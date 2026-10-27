"""Bark 3.0 add-on for Subtitld — JSON-line protocol entry point.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.

What this add-on is
-------------------
Offline English speech-to-text with Conway Research's Bark 3.0 models
(fine-tunes of NVIDIA's parakeet-tdt-0.6b-v3), on ONNX Runtime and numpy, no
torch. Serves `asr.transcribe`, with word times.

The chosen model downloads once, on its first use, from Hugging Face
(models.py), and is converted for the bundled graph (convert.py). After that
nothing is downloaded and nothing is sent anywhere.

Wire format
-----------
One JSON object per line on stdin/stdout. Stderr is free-form logging, which
the host prefixes with `[addon:bark3]` and surfaces.

Options
-------
`model` ($BARK3_MODEL): bark3 (default), bark3-plus, bark3-clean or bark3-pro.
`threads` ($BARK3_THREADS): ONNX Runtime threads; 0 picks half the logical
processors, at most 8.

Limits
------
English only. Batch only: one result per request. The audio is transcribed in
windows of about 30 seconds (bark3_addon/windows.py), with progress after
each. A cancel takes effect between windows.
"""

from __future__ import annotations

import os

# Before numpy loads: the decoder's small products are slower on many BLAS
# threads than on one (decoder.py). ONNX Runtime keeps its own threads.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from bark3_addon import ADDON_ID, PROTOCOL_VERSION, __version__, convert, download, models, tls  # noqa: E402
from bark3_addon.frontend import SAMPLE_RATE  # noqa: E402
from bark3_addon.phrases import words_to_segments  # noqa: E402
from bark3_addon.windows import windows  # noqa: E402

ADDON_VERSION = __version__
LANGUAGES = ('en',)
# Below this there is nothing to transcribe.
_MIN_SAMPLES = SAMPLE_RATE // 10
_READ_CHUNK_BYTES = 1 << 20

ERR_MODEL_MISSING = 'model_missing'
ERR_UNSUPPORTED_LANGUAGE = 'unsupported_language'
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_OOM = 'oom'
ERR_DISK_FULL = 'disk_full'
ERR_NETWORK = 'network'

log = logging.getLogger('bark3_addon')


# ---------------------------------------------------------------------------
# stdio ownership + protocol frames
# ---------------------------------------------------------------------------
_PROTO_IN = None   # binary reader for host frames
_PROTO_OUT = None  # binary writer for our frames
_proto_lock = threading.Lock()


def _claim_protocol_stdio() -> None:
    """Give the protocol exclusive use of the real stdin/stdout.

    fd-level, not just `sys.stdout = sys.stderr`: ONNX Runtime is native code
    and may write to fd 1 directly. Falls back to the Python-level swap if the fd
    dance fails.
    """
    global _PROTO_IN, _PROTO_OUT
    try:
        sys.stdout.flush()
        in_fd = os.dup(0)
        out_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull, 0)
        finally:
            os.close(devnull)
        os.dup2(2, 1)
        _PROTO_IN = os.fdopen(in_fd, 'rb')
        _PROTO_OUT = os.fdopen(out_fd, 'wb')
    except (OSError, ValueError, AttributeError):
        _PROTO_IN = sys.stdin.buffer
        _PROTO_OUT = sys.stdout.buffer
    sys.stdin = open(os.devnull, encoding='utf-8')  # stray input() -> EOFError
    sys.stdout = sys.stderr                        # stray print() -> the log


def _emit(frame: dict) -> None:
    data = (json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    out = _PROTO_OUT if _PROTO_OUT is not None else sys.stdout.buffer
    with _proto_lock:
        try:
            out.write(data)
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # host went away; the frame loop exits on the next empty read


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress',
           'value': max(0.0, min(1.0, float(value))), 'message': message})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({'id': req_id, 'type': 'error', 'code': code,
           'message': message, 'retryable': retryable})


class _AddonError(Exception):
    """An expected failure that already knows its protocol code."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class _Cancelled(Exception):
    """The host cancelled this request."""


# ---------------------------------------------------------------------------
# Options and environment
# ---------------------------------------------------------------------------
def _env(*names: str) -> str:
    """First non-empty value among `names`, stripped. '' when none are set."""
    for name in names:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value
    return ''


def _resolve_model(options: dict) -> str:
    """The request's model, else $BARK3_MODEL, else Bark 3.0. An unknown id is an error."""
    raw = str(options.get('model') or _env('BARK3_MODEL') or models.DEFAULT).strip().lower()
    if raw not in models.MODELS:
        raise _AddonError(ERR_BAD_PARAMS, f'Unknown model {raw!r}; this add-on has '
                                          f'{", ".join(models.MODELS)}.')
    return raw


def _resolve_threads(options: dict) -> int:
    raw = options.get('threads')
    if raw in (None, ''):
        raw = _env('BARK3_THREADS') or 0
    try:
        threads = int(float(raw))
    except (TypeError, ValueError):
        log.warning('threads=%r is not a number; picking automatically', raw)
        threads = 0
    if threads <= 0:
        threads = max(1, min(8, (os.cpu_count() or 2) // 2))
    return min(threads, 256)


def _resolve_language(raw) -> str:
    """'en'. The host sends BCP-47-ish tags (`en-US`); empty and 'auto' mean English too.

    Bark 3.0 transcribes English only: another language is an error, not a guess.
    """
    tag = str(raw or '').strip().lower().replace('_', '-')
    if not tag or tag in ('auto', 'und', 'mul') or tag.split('-', 1)[0] == 'en':
        return 'en'
    raise _AddonError(ERR_UNSUPPORTED_LANGUAGE,
                      f'Bark 3.0 transcribes English only, not {raw!r}; '
                      f'pick another engine for other languages.')


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str | None:
    for candidate in (_env('SUBTITLD_FFMPEG_EXECUTABLE'), _env('FFMPEG_EXECUTABLE')):
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    return shutil.which('ffmpeg')


def _no_window_startupinfo():
    """Keep ffmpeg from flashing a console window on Windows GUI hosts."""
    if sys.platform != 'win32':
        return None
    si = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    si.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return si


def _child_env() -> dict | None:
    """Environment for ffmpeg, or None to inherit ours unchanged.

    The frozen Linux bootloader prepends the bundle's `_internal/` to
    LD_LIBRARY_PATH, and a system ffmpeg that inherits it loads the bundle's
    libraries instead of its own. PyInstaller saves the host value in
    LD_LIBRARY_PATH_ORIG, so restore it for the child.
    """
    if not (getattr(sys, 'frozen', False) and sys.platform.startswith('linux')):
        return None
    env = dict(os.environ)
    original = env.get('LD_LIBRARY_PATH_ORIG')
    if original is not None:
        env['LD_LIBRARY_PATH'] = original
    else:
        env.pop('LD_LIBRARY_PATH', None)
    return env


def _decode(ffmpeg: str, src: str, cancel: threading.Event) -> np.ndarray:
    """Decode anything ffmpeg reads into 16 kHz mono float samples in [-1, 1].

    ffmpeg writes float32 straight into a pipe, so no temporary audio file is
    made and no PCM copy is kept: the samples are read in chunks into one
    buffer. stderr goes to a temporary file, so a chatty ffmpeg cannot
    fill a pipe and stall. `-nostdin` plus stdin=DEVNULL: ffmpeg polls stdin
    for keystrokes.
    """
    cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
           '-i', src, '-vn', '-sn', '-dn',
           '-ac', '1', '-ar', str(SAMPLE_RATE), '-c:a', 'pcm_f32le', '-f', 'f32le', 'pipe:1']
    data = bytearray()
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=errors, env=_child_env(),
                                startupinfo=_no_window_startupinfo())
        try:
            while True:
                if cancel.is_set():
                    raise _Cancelled()
                chunk = proc.stdout.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break
                data += chunk
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        proc.stdout.close()
        returncode = proc.wait()
        usable = len(data) // 4 * 4
        if returncode != 0 or not usable:
            errors.seek(0)
            tail = ' / '.join(errors.read().decode('utf-8', 'replace').strip().splitlines()[-5:])
            if 'No space left' in tail:
                raise _AddonError(ERR_DISK_FULL, f'Out of disk space while decoding the audio: {tail}')
            raise _AddonError(ERR_INTERNAL,
                              f'Could not decode the audio with ffmpeg '
                              f'(exit {returncode}): {tail or "no output"}')
    # ffmpeg wrote little-endian floats.
    return np.frombuffer(memoryview(data)[:usable], '<f4').astype(np.float32)


# ---------------------------------------------------------------------------
# The model: downloaded and converted once, loaded on first use
# ---------------------------------------------------------------------------
_engine = None
_engine_key: tuple | None = None
_engine_lock = threading.Lock()


def _megabytes(count: int) -> str:
    return f'{count / 1e6:,.0f} MB'


def model_folder(model: str) -> Path:
    """Where a model's converted files go: beside its download, under the pinned revision."""
    return download.local_path(models.pins()[model]).parent / 'onnx'


def _prepare(state: '_Request', model: str) -> Path:
    """Download (once) and convert (once). Progress 0.05 -> 0.40."""
    pins = models.pins()
    model_pin, tokenizer_pin = pins[model], pins['tokenizer']
    name = models.MODELS[model]['name']
    folder = model_folder(model)
    needed = [tokenizer_pin] + ([] if convert.is_ready(folder) else [model_pin])
    todo = download.missing(needed)
    if todo:
        total = sum(pin['size'] for pin in todo)
        message = f'Downloading {name} ({_megabytes(total)}, once)...'
        _emit_progress(state.req_id, 0.05, message)
        last = [0.0]

        def progress(done: int, total_bytes: int) -> None:
            now = time.monotonic()
            if now - last[0] >= 0.25 or done >= total_bytes:
                last[0] = now
                _emit_progress(state.req_id, 0.05 + 0.2 * done / max(1, total_bytes), message)

        started = time.monotonic()
        try:
            download.ensure(needed, progress, state.cancel.is_set)
        except download.Cancelled:
            raise _Cancelled() from None
        except download.DownloadError as exc:
            raise _AddonError(ERR_NETWORK, f'Could not download {name}: {exc}', retryable=True) from exc
        log.info('downloaded %s in %.1fs', _megabytes(total), time.monotonic() - started)
    if not convert.is_ready(folder):
        message = f'Preparing {name} for this computer (once)...'
        _emit_progress(state.req_id, 0.25, message)

        def converted(done: int, total: int) -> None:
            _emit_progress(state.req_id, 0.25 + 0.15 * done / max(1, total), message)

        started = time.monotonic()
        source = download.local_path(model_pin)
        try:
            convert.build(source, folder, converted, state.cancel.is_set)
        except convert.Cancelled:
            raise _Cancelled() from None
        except OSError as exc:
            if getattr(exc, 'errno', None) == 28:
                raise _AddonError(ERR_DISK_FULL, f'Out of disk space while preparing {name}: {exc}') from exc
            raise
        log.info('converted %s in %.1fs', name, time.monotonic() - started)
        # The download is not read again; the converted folder replaces it.
        for path in (source, source.with_name(source.name + '.ok')):
            try:
                path.unlink()
            except OSError:
                pass
    return folder


def _load_engine(state: '_Request', model: str, threads: int):
    """The engine for `model`, downloading, converting and loading as needed. One at a time."""
    global _engine, _engine_key
    from bark3_addon.engine import Engine  # onnxruntime: only once a request comes

    with _engine_lock:
        if _engine is not None and _engine_key == (model, threads):
            return _engine
        folder = _prepare(state, model)
        if state.cancel.is_set():
            raise _Cancelled()
        name = models.MODELS[model]['name']
        _emit_progress(state.req_id, 0.4, f'Loading {name}...')
        _engine = _engine_key = None   # one model in memory at a time
        started = time.monotonic()
        try:
            _engine = Engine(folder, download.local_path(models.pins()['tokenizer']), threads,
                             models.MODELS[model]['pad_sec'])
        except Exception as exc:  # noqa: BLE001 — ONNX Runtime raises its own types
            raise _AddonError(ERR_MODEL_MISSING, f'{name} failed to load from {folder}: {exc}. '
                                                 f'Delete that folder to prepare it again.') from exc
        _engine_key = (model, threads)
        log.info('loaded %s with %d threads in %.1fs', name, threads, time.monotonic() - started)
        return _engine


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
class _Request:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None


_active: dict[str, _Request] = {}
_active_lock = threading.Lock()


def _handle_transcribe(state: _Request, params: dict) -> None:
    req_id = state.req_id
    options = params.get('options')
    if not isinstance(options, dict):
        options = {}
    audio_path = params.get('audio_path')
    if not isinstance(audio_path, str) or not audio_path.strip():
        _emit_error(req_id, ERR_BAD_PARAMS, 'audio_path is required')
        return
    if not os.path.isfile(audio_path):
        _emit_error(req_id, ERR_BAD_PARAMS, f'audio_path is not a file: {audio_path!r}')
        return
    try:
        language = _resolve_language(params.get('language'))
        model = _resolve_model(options)
    except _AddonError as exc:
        _emit_error(req_id, exc.code, exc.message)
        return
    threads = _resolve_threads(options)
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        _emit_error(req_id, ERR_BAD_PARAMS,
                    'ffmpeg not found. Set SUBTITLD_FFMPEG_EXECUTABLE or put ffmpeg on PATH.')
        return

    outcome: tuple = ('error', ERR_INTERNAL, 'no outcome', False)
    try:
        log.info('transcribe %s: model=%s threads=%d', req_id, model, threads)
        if state.cancel.is_set():
            raise _Cancelled()

        # 1. Decode: 0.00 -> 0.05
        _emit_progress(req_id, 0.0, 'Decoding audio...')
        samples = _decode(ffmpeg, audio_path, state.cancel)
        duration = len(samples) / float(SAMPLE_RATE)
        if state.cancel.is_set():
            raise _Cancelled()
        _emit_progress(req_id, 0.05, 'Decoded audio')

        if len(samples) < _MIN_SAMPLES:
            log.info('transcribe %s: %.3fs of audio, nothing to transcribe', req_id, duration)
            segments: list[dict] = []
        else:
            # 2. Model: 0.05 -> 0.45, then window by window: 0.45 -> 1.00
            engine = _load_engine(state, model, threads)
            if state.cancel.is_set():
                raise _Cancelled()
            _emit_progress(req_id, 0.45, 'Transcribing...')
            started = time.monotonic()
            ranges = windows(samples)
            words: list[dict] = []
            for begin, end in ranges:
                if end - begin >= _MIN_SAMPLES:
                    words.extend(engine.transcribe(samples[begin:end], begin / float(SAMPLE_RATE),
                                                   state.cancel.is_set))
                if state.cancel.is_set():
                    raise _Cancelled()
                _emit_progress(req_id, 0.45 + 0.55 * end / len(samples), 'Transcribing...')
            log.info('transcribe %s: %.1fs of audio in %d windows, %.2fs', req_id, duration,
                     len(ranges), time.monotonic() - started)
            segments = words_to_segments(words, duration)
        if state.cancel.is_set():
            raise _Cancelled()
        outcome = ('result', {
            'segments': segments,
            'duration_sec': round(duration, 3),
            'model': model,
            'language': language,
        })
    except _Cancelled:
        log.info('transcribe %s: cancelled', req_id)
        outcome = ('error', ERR_CANCELLED, 'cancelled', False)
    except _AddonError as exc:
        log.warning('transcribe %s failed: [%s] %s', req_id, exc.code, exc.message)
        outcome = ('error', exc.code, exc.message, exc.retryable)
    except MemoryError:
        outcome = ('error', ERR_OOM, 'Out of memory while transcribing. Close other programs and try again.', False)
    except Exception as exc:  # noqa: BLE001 — every path must end in one terminal frame
        log.exception('transcribe %s failed', req_id)
        outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)

    if outcome[0] == 'result':
        _emit_progress(req_id, 1.0, '')
        _emit_result(req_id, outcome[1])
    else:
        _emit_error(req_id, *outcome[1:])


def _spawn(req_id: str, target, *args) -> None:
    state = _Request(req_id)
    with _active_lock:
        _active[req_id] = state

    def _run():
        try:
            target(state, *args)
        except BaseException:  # noqa: BLE001
            log.exception('worker crashed')
            tail = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tail}')
        finally:
            with _active_lock:
                _active.pop(req_id, None)

    thread = threading.Thread(target=_run, name=f'bark3-{req_id}', daemon=True)
    state.thread = thread
    thread.start()


# ---------------------------------------------------------------------------
# Handshake + frame loop
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [{
            'task': 'asr.transcribe',
            'languages': list(LANGUAGES),
            # Batch transcription only: the live `asr.stream` task is not offered.
            'streaming': False,
            'voice_clone': False,
        }],
    })


def _read_frame() -> dict | None:
    source = _PROTO_IN if _PROTO_IN is not None else sys.stdin.buffer
    line = source.readline()
    if not line:
        return None  # stdin closed: the host went away
    text = line.decode('utf-8', errors='replace').strip().lstrip('﻿')
    if not text:
        return {}
    try:
        frame = json.loads(text)
    except ValueError as exc:
        log.warning('malformed frame %r: %s', line[:200], exc)
        return {}
    return frame if isinstance(frame, dict) else {}


def _main_loop() -> int:
    while True:
        frame = _read_frame()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'shutdown':
            log.info('shutdown requested')
            return 0

        if ftype == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            continue

        if ftype == 'cancel':
            target = frame.get('target')
            with _active_lock:
                state = _active.get(target) if isinstance(target, str) else None
            if state is not None:
                log.info('cancel %s', target)
                state.cancel.set()
            continue

        if ftype == 'asr.transcribe':
            req_id = frame.get('id')
            params = frame.get('params')
            if not isinstance(req_id, str) or not req_id:
                log.warning('asr.transcribe without id')
                continue
            if params is None:
                params = {}
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            _spawn(req_id, _handle_transcribe, params)
            continue

        # Unknown task: still answer, so the host's future resolves instead of
        # hanging until its timeout.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def _stop_workers(budget_sec: float = 3.5) -> bool:
    """Cancel and join every worker. True when none is left running.

    The host allows 5 s between `shutdown` and SIGTERM, so stay under it.
    """
    with _active_lock:
        states = list(_active.values())
    for state in states:
        state.cancel.set()
    deadline = time.monotonic() + budget_sec
    for state in states:
        if state.thread is not None:
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(s.thread is not None and s.thread.is_alive() for s in states)


def _tiny_graph() -> bytes:
    """A serialized ONNX model computing x + x, written without the onnx package."""
    def varint(n: int) -> bytes:
        out = b''
        while True:
            byte, n = n & 0x7F, n >> 7
            out += bytes([byte | (0x80 if n else 0)])
            if not n:
                return out

    def field(number: int, payload: bytes) -> bytes:  # length-delimited
        return varint(number << 3 | 2) + varint(len(payload)) + payload

    def number(field_no: int, value: int) -> bytes:   # varint
        return varint(field_no << 3) + varint(value)

    node = field(1, b'x') + field(1, b'x') + field(2, b'y') + field(4, b'Add')
    dim = field(1, number(1, 3))                                  # TensorShapeProto.Dimension
    tensor = number(1, 1) + field(2, dim)                         # elem_type FLOAT, shape
    value_type = field(1, tensor)                                 # TypeProto.tensor_type

    def value_info(name: bytes) -> bytes:
        return field(1, name) + field(2, value_type)

    graph = field(1, node) + field(2, b'g') + field(11, value_info(b'x')) + field(12, value_info(b'y'))
    opset = field(1, b'') + number(2, 13)
    return number(1, 8) + field(7, graph) + field(8, opset)


def _self_test() -> int:
    """Check what the frozen bundle needs, without the network or a model.

    ONNX Runtime must load and run a graph, constriction must decode an ANS
    stream, the front end must run, the bundled graph and its layout must be
    there and agree, and the pins, TLS and hashing the downloads use.
    """
    import ssl

    import constriction
    import onnxruntime as ort

    from bark3_addon import frontend

    session = ort.InferenceSession(_tiny_graph(), providers=['CPUExecutionProvider'])
    doubled = session.run(None, {'x': np.ones(3, dtype=np.float32)})[0]
    message = np.array([3, 1, 4, 1, 5, 2], np.int32)
    probabilities = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    coder = constriction.stream.stack.AnsCoder()
    model = constriction.stream.model.Categorical(probabilities, perfect=False)
    coder.encode_reverse(message, model)
    words_back = constriction.stream.stack.AnsCoder(coder.get_compressed()).decode(model, len(message))
    tone = (0.1 * np.sin(np.arange(SAMPLE_RATE) * 2 * np.pi * 440 / SAMPLE_RATE)).astype(np.float32)
    features = frontend.features(tone)
    graph = convert.GRAPH_DIR / convert.GRAPH_FILE
    graph_ok = graph.is_file() and \
        hashlib.sha256(graph.read_bytes()).hexdigest() == convert.layout()['graph_sha256']
    ok = (bool(np.allclose(doubled, 2)) and bool(np.array_equal(words_back, message))
          and features.shape == (101, 128) and bool(np.isfinite(features).all()) and graph_ok
          and set(models.pins()) == set(models.MODELS) | {'tokenizer'})
    report = {
        'ok': ok,
        'version': ADDON_VERSION,
        'onnxruntime': ort.__version__,
        'numpy': np.__version__,
        'constriction': getattr(constriction, '__version__', ''),
        'graph': graph_ok,
        'pins': len(models.pins()),
        'openssl': ssl.OPENSSL_VERSION,
        'sha256': hashlib.sha256(b'').hexdigest()[:8],
        'ca_bundle': os.environ.get('SSL_CERT_FILE', ''),
    }
    print(json.dumps(report), file=sys.__stdout__, flush=True)
    return 0 if ok else 1


def main() -> int:
    if '--version' in sys.argv[1:]:
        print(f'bark3-addon {ADDON_VERSION}')
        return 0
    logging.basicConfig(
        level=(_env('BARK3_ADDON_LOG_LEVEL') or 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    tls.configure_ca_bundle()
    if '--self-test' in sys.argv[1:]:
        return _self_test()
    _claim_protocol_stdio()
    log.info('bark3-addon %s on %s %s, python %s',
             ADDON_VERSION, platform.system(), platform.machine(), platform.python_version())
    if _find_ffmpeg() is None:
        # Not fatal: better to appear in the engine list and fail the first
        # request with a clear message than to vanish from the UI.
        log.warning('ffmpeg not found; requests will fail until one is available')

    _send_hello()
    rc = 1
    try:
        rc = _main_loop()
    except KeyboardInterrupt:
        rc = 130
    except Exception:  # noqa: BLE001
        log.exception('main loop crashed')
        rc = 1
    finally:
        stopped = _stop_workers()
    if not stopped:
        # A worker is still inside ONNX Runtime. Normal teardown would free its
        # session under it, so leave without teardown.
        log.warning('a transcription is still running; exiting without cleanup')
        try:
            sys.stderr.flush()
        finally:
            os._exit(rc)
    return rc


if __name__ == '__main__':
    sys.exit(main())
