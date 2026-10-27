"""The downloader against a local HTTP server: checks, resumes and cancels.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import hashlib
import http.server
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bark3_addon import download  # noqa: E402

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB
REPO = 'ConwayResearch/Underdog-Bark-3.0'


class _Handler(http.server.BaseHTTPRequestHandler):
    requests: list[str] = []
    paths: list[str] = []

    def do_GET(self):  # noqa: N802
        type(self).requests.append(self.headers.get('Range') or '')
        type(self).paths.append(self.path)
        start = 0
        if self.headers.get('Range'):
            start = int(self.headers['Range'].split('=')[1].split('-')[0])
            self.send_response(206)
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture()
def server(monkeypatch, tmp_path):
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('HF_ENDPOINT', f'http://127.0.0.1:{httpd.server_address[1]}')
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    _Handler.requests = []
    _Handler.paths = []
    yield httpd
    httpd.shutdown()


def _pin(sha256: str | None = None) -> dict:
    return {'path': 'model.bark', 'repo': REPO, 'revision': 'a' * 40,
            'size': len(PAYLOAD), 'sha256': sha256 or hashlib.sha256(PAYLOAD).hexdigest()}


def test_downloads_checks_and_stamps(server, tmp_path):
    pin = _pin()
    seen = []
    download.ensure([pin], lambda done, total: seen.append((done, total)), lambda: False)
    path = download.local_path(pin)
    assert path == (tmp_path / 'bark3' / 'ConwayResearch--Underdog-Bark-3.0' / ('a' * 40)
                    / 'model.bark')
    assert path.read_bytes() == PAYLOAD
    assert download.is_present(pin)
    assert seen[-1] == (len(PAYLOAD), len(PAYLOAD))
    # Present now: nothing is fetched again.
    download.ensure([pin], lambda *_: None, lambda: False)
    assert _Handler.requests == ['']
    assert _Handler.paths == [f'/{REPO}/resolve/{"a" * 40}/model.bark']


def test_resumes_from_the_part_file(server):
    pin = _pin()
    part = download.local_path(pin).with_name('model.bark.part')
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:1000])
    download.ensure([pin], lambda *_: None, lambda: False)
    assert _Handler.requests == ['bytes=1000-']
    assert download.local_path(pin).read_bytes() == PAYLOAD


def test_a_wrong_sha256_discards_the_file(server):
    pin = _pin(sha256='0' * 64)
    with pytest.raises(download.DownloadError, match='sha256'):
        download.ensure([pin], lambda *_: None, lambda: False)
    path = download.local_path(pin)
    assert not path.exists() and not path.with_name(path.name + '.part').exists()


def test_a_cancel_stops_the_download_and_keeps_the_part(server):
    pin = _pin()
    with pytest.raises(download.Cancelled):
        download.ensure([pin], lambda *_: None, lambda: True)
    assert not download.is_present(pin)


def test_a_file_without_its_stamp_is_checked_not_fetched(server):
    pin = _pin()
    path = download.local_path(pin)
    path.parent.mkdir(parents=True)
    path.write_bytes(PAYLOAD)
    assert not download.is_present(pin)
    download.ensure([pin], lambda *_: None, lambda: False)
    assert download.is_present(pin)
    assert _Handler.requests == []


def test_an_unreachable_server_is_a_download_error(monkeypatch, tmp_path):
    monkeypatch.setenv('HF_ENDPOINT', 'http://127.0.0.1:9')  # discard port: nothing listens
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    with pytest.raises(download.DownloadError):
        download.ensure([_pin()], lambda *_: None, lambda: False)
