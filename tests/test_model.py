"""Real transcription with Bark 3.0, downloaded and converted like a user's first run.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The first run downloads 163 MB into $SUBTITLD_MODELS_DIR (default: the user's
model cache) and converts it, about half a minute. Later runs reuse it.
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

pytestmark = pytest.mark.model


def _transcribe(tmp_path, params: dict, cancel_on: str | None = None) -> tuple[dict, list[dict]]:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), SUBTITLD_TEMP_DIR=str(tmp_path))
    proc = subprocess.Popen([sys.executable, '-m', 'bark3_addon'], cwd=tmp_path, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    progress = []
    try:
        assert json.loads(proc.stdout.readline())['type'] == 'hello'
        proc.stdin.write((json.dumps({'id': 'r', 'type': 'asr.transcribe', 'params': params}) + '\n').encode())
        proc.stdin.flush()
        while True:
            frame = json.loads(proc.stdout.readline())
            if frame['type'] == 'progress':
                progress.append(frame)
                if cancel_on and frame['message'] == cancel_on:
                    proc.stdin.write(b'{"type": "cancel", "target": "r"}\n')
                    proc.stdin.flush()
                    cancel_on = None
                continue
            return frame, progress
    finally:
        proc.stdin.write(b'{"type": "shutdown"}\n')
        proc.stdin.flush()
        assert proc.wait(timeout=60) == 0


def test_transcribes_with_word_times(tmp_path):
    frame, progress = _transcribe(tmp_path, {'audio_path': str(SAMPLE), 'language': 'en'})
    assert frame['type'] == 'result', frame
    data = frame['data']
    assert data['model'] == 'bark3' and data['language'] == 'en'
    assert 10.9 < data['duration_sec'] < 11.1
    text = ' '.join(row['text'] for row in data['segments'])
    assert text == ('And so, my fellow Americans, ask not what your country can do for you. '
                    'Ask what you can do for your country')
    starts = [row['start'] for row in data['segments']]
    assert starts == sorted(starts) and 0 <= starts[0] < 1.0
    assert data['segments'][-1]['end'] <= data['duration_sec'] + 0.1
    values = [p['value'] for p in progress]
    assert values == sorted(values) and values[-1] == 1.0


def test_a_cancel_while_transcribing_ends_the_request(tmp_path):
    long_audio = tmp_path / 'long.flac'
    subprocess.run(['ffmpeg', '-v', 'error', '-stream_loop', '9', '-i', str(SAMPLE), str(long_audio)],
                   check=True)
    frame, _ = _transcribe(tmp_path, {'audio_path': str(long_audio)}, cancel_on='Transcribing...')
    assert frame['type'] == 'error' and frame['code'] == 'cancelled'
