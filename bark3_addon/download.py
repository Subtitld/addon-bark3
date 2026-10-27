"""Fetch pinned files from Hugging Face: resumable, checked by size and sha256.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Each file lands in the host's model cache under its repository and revision,
so files of different releases never overwrite each other:

    $SUBTITLD_MODELS_DIR/bark3/<owner>--<name>/<revision>/<path in the repository>

A file is downloaded to `<name>.part` and renamed into place only once its size
and sha256 match the pin; an interrupted download resumes from the `.part`
on the next request. A file in place is trusted once a `.ok` stamp beside it
says it was checked; the stamp holds the sha256, so a re-pinned file is
checked again. $HF_ENDPOINT points the downloads at a mirror.
"""

from __future__ import annotations

import hashlib
import logging
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from bark3_addon import __version__

log = logging.getLogger('bark3_addon.download')

_CHUNK = 1 << 20


class Cancelled(Exception):
    """The request was cancelled."""


class DownloadError(Exception):
    """A file could not be fetched, or did not check out."""


def cache_root() -> Path:
    """$SUBTITLD_MODELS_DIR/bark3, else ~/.cache/subtitld/models/bark3."""
    base = (os.environ.get('SUBTITLD_MODELS_DIR') or '').strip()
    root = Path(os.path.expanduser(base)) if base else Path.home() / '.cache' / 'subtitld' / 'models'
    return root / 'bark3'


def local_path(pin: dict) -> Path:
    """Where a pin (a pins.json entry) goes."""
    return cache_root() / pin['repo'].replace('/', '--') / pin['revision'] / Path(*pin['path'].split('/'))


def _stamp(path: Path) -> Path:
    return path.with_name(path.name + '.ok')


def is_present(pin: dict) -> bool:
    """True when the file is in place and was checked against this pin."""
    path = local_path(pin)
    try:
        return (path.is_file() and path.stat().st_size == pin['size']
                and _stamp(path).read_text(encoding='ascii').strip() == pin['sha256'])
    except OSError:
        return False


def missing(pins: list[dict]) -> list[dict]:
    return [pin for pin in pins if not is_present(pin)]


def _ssl_context() -> ssl.SSLContext:
    # bark3_addon.tls has set SSL_CERT_FILE for the frozen Linux build;
    # elsewhere this is the platform's own trust store.
    return ssl.create_default_context()


def _sha256(path: Path, cancel: Callable[[], bool]) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(_CHUNK):
            if cancel():
                raise Cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(pin: dict, on_bytes: Callable[[int], None], cancel: Callable[[], bool]) -> None:
    dest = local_path(pin)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    size = pin['size']
    have = part.stat().st_size if part.is_file() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        endpoint = (os.environ.get('HF_ENDPOINT') or 'https://huggingface.co').rstrip('/')
        url = f'{endpoint}/{pin["repo"]}/resolve/{pin["revision"]}/{pin["path"]}'
        headers = {'User-Agent': f'subtitld-bark3-addon/{__version__}'}
        if have:
            headers['Range'] = f'bytes={have}-'
        log.info('fetching %s (%s bytes%s)', url, f'{size:,}', f', resuming at {have:,}' if have else '')
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=60, context=_ssl_context()) as response:
                if have and response.status != 206:
                    have = 0  # the server ignored the range: start over
                on_bytes(have)
                with part.open('ab' if have else 'wb') as out:
                    while chunk := response.read(_CHUNK):
                        if cancel():
                            raise Cancelled()
                        out.write(chunk)
                        on_bytes(len(chunk))
        except urllib.error.HTTPError as exc:
            raise DownloadError(f'{dest.name}: HTTP {exc.code} from {url}') from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise DownloadError(f'{dest.name}: {getattr(exc, "reason", exc)}') from exc
    else:
        on_bytes(have)
    got = part.stat().st_size
    if got != size:
        raise DownloadError(f'{dest.name}: got {got:,} bytes, expected {size:,}. Try again to resume.')
    if _sha256(part, cancel) != pin['sha256']:
        part.unlink()
        raise DownloadError(f'{dest.name} does not match its pinned sha256; the download was discarded.')
    os.replace(part, dest)
    _stamp(dest).write_text(pin['sha256'] + '\n', encoding='ascii')


def ensure(pins: list[dict], progress: Callable[[int, int], None],
           cancel: Callable[[], bool]) -> None:
    """Download whichever of `pins` is missing. `progress(done, total)` is in bytes."""
    todo = missing(pins)
    if not todo:
        return
    total = sum(pin['size'] for pin in todo)
    done = 0

    def on_bytes(count: int) -> None:
        nonlocal done
        done += count
        progress(min(done, total), total)

    for pin in todo:
        dest = local_path(pin)
        if dest.is_file() and dest.stat().st_size == pin['size'] and _sha256(dest, cancel) == pin['sha256']:
            _stamp(dest).write_text(pin['sha256'] + '\n', encoding='ascii')  # checked, stamp lost
            on_bytes(pin['size'])
            continue
        _fetch(pin, on_bytes, cancel)
