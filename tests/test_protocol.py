"""The add-on spoken over its wire format, in a real process.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

These run `python -m bark3_addon` and talk JSON lines to it, the way the host
does. They cover every answer the add-on owes the host without a model or the
network: downloads point at a port nothing listens on.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / 'tests' / 'data' / 'jfk.flac'


@pytest.fixture()
def addon(tmp_path):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    env['SUBTITLD_MODELS_DIR'] = str(tmp_path / 'models')
    env['HF_ENDPOINT'] = 'http://127.0.0.1:9'  # nothing listens: no test reaches the network
    env.pop('BARK3_MODEL', None)
    proc = subprocess.Popen([sys.executable, '-m', 'bark3_addon'], cwd=tmp_path, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    hello = _recv(proc)
    _send(proc, {'type': 'ready', 'host': 'subtitld', 'host_version': 'test', 'protocol': 1})
    yield proc, hello
    try:
        _send(proc, {'type': 'shutdown'})
        assert proc.wait(timeout=15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()


def _send(proc, frame: dict) -> None:
    proc.stdin.write((json.dumps(frame) + '\n').encode('utf-8'))
    proc.stdin.flush()


def _recv(proc) -> dict:
    line = proc.stdout.readline()
    assert line, 'the add-on closed its output: ' + proc.stderr.read().decode('utf-8', 'replace')
    return json.loads(line)


def _terminal(proc, req_id: str) -> dict:
    while True:
        frame = _recv(proc)
        if frame.get('id') == req_id and frame['type'] in ('result', 'error'):
            return frame


def test_hello_declares_english_transcription(addon):
    _, hello = addon
    assert hello['type'] == 'hello' and hello['addon'] == 'bark3' and hello['protocol'] == 1
    (capability,) = hello['capabilities']
    assert capability['task'] == 'asr.transcribe'
    assert capability['languages'] == ['en']
    assert capability['streaming'] is False


def test_an_unknown_task_is_still_answered(addon):
    proc, _ = addon
    _send(proc, {'id': 'x', 'type': 'tts.synthesize', 'params': {}})
    frame = _terminal(proc, 'x')
    assert frame['type'] == 'error' and frame['code'] == 'bad_params'


@pytest.mark.parametrize('params, code', [
    ({}, 'bad_params'),
    ({'audio_path': '/no/such/file.wav'}, 'bad_params'),
    ({'audio_path': str(SAMPLE), 'language': 'pt-BR'}, 'unsupported_language'),
    ({'audio_path': str(SAMPLE), 'options': {'model': 'bark2'}}, 'bad_params'),
])
def test_bad_requests_are_refused(addon, params, code):
    proc, _ = addon
    _send(proc, {'id': 'r', 'type': 'asr.transcribe', 'params': params})
    frame = _terminal(proc, 'r')
    assert frame['type'] == 'error' and frame['code'] == code, frame


def test_no_network_is_a_retryable_network_error(addon):
    proc, _ = addon
    _send(proc, {'id': 'n', 'type': 'asr.transcribe',
                 'params': {'audio_path': str(SAMPLE), 'language': 'en-US'}})
    messages = []
    while True:
        frame = _recv(proc)
        if frame['type'] == 'progress':
            messages.append(frame['message'])
            continue
        break
    assert frame['type'] == 'error' and frame['code'] == 'network' and frame['retryable'] is True
    assert any(m.startswith('Downloading Bark 3.0 (165 MB') for m in messages), messages


def test_self_test_passes():
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    out = subprocess.run([sys.executable, '-m', 'bark3_addon', '--self-test'], capture_output=True,
                         text=True, timeout=120, env=env, cwd=REPO_ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1])['ok'] is True
