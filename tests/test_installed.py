"""The frozen binary, unpacked from its release archive the way Subtitld installs it.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

$BARK3_TEST_BINARY names the binary; without it these tests skip (and the
release workflow refuses skipped tests). With nothing but what is in its own
bundle, the binary downloads Bark 3.0 (163 MB) into $SUBTITLD_MODELS_DIR,
converts it, transcribes a clip and cancels a longer one.

    BARK3_TEST_BINARY=dist/bark3-addon/bark3-addon pytest tests/test_installed.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import wave
from pathlib import Path

import pytest

BINARY = os.environ.get('BARK3_TEST_BINARY', '')
SAMPLE = Path(__file__).resolve().parent / 'data' / 'jfk.flac'
pytestmark = pytest.mark.skipif(not BINARY, reason='BARK3_TEST_BINARY is not set')


class Addon:
    """The binary on the protocol, with a watchdog so a hang fails instead of stalling CI."""

    def __init__(self, tmp_path: Path, timeout: float, **options: str):
        env = dict(os.environ, SUBTITLD_TEMP_DIR=str(tmp_path), **options)
        env.setdefault('SUBTITLD_MODELS_DIR', str(tmp_path / 'models'))
        env.setdefault('BARK3_ADDON_LOG_LEVEL', 'DEBUG')
        self.log_path = tmp_path / 'addon.log'
        self.log = open(self.log_path, 'wb')
        self.proc = subprocess.Popen([BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, env=env)
        self.timer = threading.Timer(timeout, self.proc.kill)
        self.timer.start()

    def send(self, frame: dict) -> None:
        self.proc.stdin.write((json.dumps(frame) + '\n').encode())
        self.proc.stdin.flush()

    def log_tail(self, lines: int = 60) -> str:
        """The end of the add-on's stderr, for a failure message."""
        self.log.flush()
        text = self.log_path.read_text(encoding='utf-8', errors='replace')
        return '\n'.join(text.splitlines()[-lines:])

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        assert line, 'the add-on closed its output (crashed or timed out):\n' + self.log_tail()
        return json.loads(line)

    def until_terminal(self, req_id: str, on_progress=None) -> dict:
        while True:
            frame = self.read()
            if frame.get('id') != req_id:
                continue
            if frame['type'] in ('result', 'error'):
                return frame
            if on_progress is not None:
                on_progress(frame)

    def close(self) -> int:
        try:
            self.send({'type': 'shutdown'})
        except (BrokenPipeError, OSError):
            pass
        try:
            return self.proc.wait(timeout=15)
        finally:
            self.timer.cancel()
            self.proc.kill()
            self.log.close()


def _long_wav(path: Path, repeats: int) -> None:
    """The sample, `repeats` times, as a WAV the stdlib writes (ffmpeg is not needed to make it)."""
    ffmpeg = shutil.which('ffmpeg')
    assert ffmpeg, 'the tests need ffmpeg, as the add-on does'
    pcm = subprocess.run([ffmpeg, '-v', 'error', '-i', str(SAMPLE), '-ac', '1', '-ar', '16000',
                          '-f', 's16le', 'pipe:1'], capture_output=True, check=True).stdout
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(pcm * repeats)


def test_version():
    out = subprocess.run([BINARY, '--version'], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.startswith('bark3-addon ')


def test_self_test_loads_everything():
    out = subprocess.run([BINARY, '--self-test'], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report['ok'] is True and report['graph'] is True and report['pins'] == 5


def test_downloads_converts_transcribes_then_cancels(tmp_path):
    addon = Addon(tmp_path, timeout=1800)
    try:
        hello = addon.read()
        assert hello['type'] == 'hello' and hello['addon'] == 'bark3'
        addon.send({'type': 'ready', 'host': 'test'})

        messages = []
        addon.send({'id': 'a', 'type': 'asr.transcribe',
                    'params': {'audio_path': str(SAMPLE), 'language': 'en-US'}})
        result = addon.until_terminal('a', lambda f: messages.append(f['message']))
        assert result['type'] == 'result', f'{result}\n{addon.log_tail()}'
        text = ' '.join(row['text'] for row in result['data']['segments'])
        assert 'ask not what your country can do for you' in text.lower(), \
            f'{result}\n{addon.log_tail()}'
        assert any(m.startswith('Downloading Bark 3.0') for m in messages)
        assert any(m.startswith('Preparing Bark 3.0') for m in messages)

        long_audio = tmp_path / 'long.wav'
        _long_wav(long_audio, 12)
        cancelled = []

        def on_progress(frame):
            if frame.get('message') == 'Transcribing...' and not cancelled:
                cancelled.append(frame['value'])
                addon.send({'type': 'cancel', 'target': 'b'})

        addon.send({'id': 'b', 'type': 'asr.transcribe', 'params': {'audio_path': str(long_audio)}})
        error = addon.until_terminal('b', on_progress)
        assert cancelled and error['type'] == 'error' and error['code'] == 'cancelled'
    finally:
        assert addon.close() == 0
