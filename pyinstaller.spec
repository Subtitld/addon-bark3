# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld Bark 3.0 add-on.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Produces `dist/bark3-addon/` (launcher + `_internal/`): ONNX Runtime, numpy,
constriction, the add-on and the encoder's graph without weights. The models
are not in it; the chosen one downloads on its first use. The release step copies the folder to `bin/` inside the
archive, next to manifest.json.

Run from the add-on root:

    pyinstaller pyinstaller.spec --noconfirm
    dist/bark3-addon/bark3-addon --self-test
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SPEC_ROOT = Path(SPECPATH).resolve()
PACKAGE = SPEC_ROOT / 'bark3_addon'

# Fail the build, loudly, if the pins or the graph are missing. Otherwise the
# frozen add-on would answer the handshake and then fail the first request.
_FILES = [PACKAGE / 'pins.json', PACKAGE / 'graph' / 'encoder.onnx', PACKAGE / 'graph' / 'layout.json']
_MISSING = [str(p) for p in _FILES if not p.is_file()]
if _MISSING:
    raise SystemExit(f'FATAL: missing {", ".join(_MISSING)}. Run tools/pin_models.py and tools/build_graph.py.')

datas = [
    (str(PACKAGE / 'pins.json'), 'bark3_addon'),
    (str(PACKAGE / 'graph'), 'bark3_addon/graph'),
]


# ---------------------------------------------------------------------------
# libexpat sanity check (from addon-whispercpp).
#
# PyInstaller bundles whatever expat it resolved on the build host, for
# CPython's pyexpat. If that host has an OLD expat shadowing the distro one
# (for example a hand-built 2.5.0 in /usr/local/lib), the frozen binary dies
# before reaching Python:
#     Failed to load libpython3.x.so: undefined symbol: XML_SetReparseDeferralEnabled
# So verify the bundled copy exports the symbol libpython needs, and substitute
# a known-good system copy if it does not.
# ---------------------------------------------------------------------------
_EXPAT_SYMBOL = 'XML_SetReparseDeferralEnabled'


def _exports_symbol(path: str, symbol: str) -> bool:
    try:
        out = subprocess.run(['nm', '-D', '--defined-only', path],
                             capture_output=True, text=True, timeout=30)
        return symbol in out.stdout
    except Exception:
        return True  # no nm (macOS/Windows): assume fine rather than break the build


def _fix_libexpat(entries):
    if not sys.platform.startswith('linux'):
        return entries
    out = []
    for entry in entries:
        name = entry[0] if entry else ''
        src = entry[1] if len(entry) > 1 else ''
        if not os.path.basename(name).startswith('libexpat.so') or _exports_symbol(src, _EXPAT_SYMBOL):
            out.append(entry)
            continue
        for cand in ('/usr/lib/x86_64-linux-gnu/libexpat.so.1', '/usr/lib64/libexpat.so.1',
                     '/lib/x86_64-linux-gnu/libexpat.so.1'):
            if os.path.exists(cand) and _exports_symbol(cand, _EXPAT_SYMBOL):
                print(f'WARNING: {src} lacks {_EXPAT_SYMBOL}; substituting {cand}')
                out.append((name, cand) + tuple(entry[2:]))
                break
        else:
            print(f'ERROR: bundled libexpat lacks {_EXPAT_SYMBOL}; the frozen binary will not start.')
            out.append(entry)
    return out


a = Analysis(
    [str(PACKAGE / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=[],
    datas=datas,
    # Imported inside functions, so the handshake does not wait for ONNX Runtime.
    hiddenimports=['bark3_addon.engine', 'bark3_addon.decoder', 'bark3_addon.tokens'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # ONNX Runtime's optional parts: training, the converters and tools,
        # and what only they import. Inference needs none of it.
        'onnxruntime.training', 'onnxruntime.tools', 'onnxruntime.transformers',
        'onnxruntime.quantization', 'onnx', 'sympy', 'coloredlogs', 'humanfriendly',
        'torch', 'scipy', 'matplotlib', 'pandas', 'IPython',
        'PySide6', 'PyQt6', 'PyQt5', 'tkinter', 'Tkinter', '_tkinter',
        'pytest', '_pytest', 'pluggy', 'iniconfig', 'setuptools', 'pkg_resources',
        'unittest', 'pydoc_data',
        # Standard modules the add-on never calls, reached only through
        # optional imports. Each would bring a native library, and with it a
        # licence to ship. ssl and hashlib stay: the model downloads over
        # HTTPS and is checked by sha256. lzma stays: .bark v3 files compress
        # their side data with it (unpack.py).
        'bz2', '_bz2', 'compression', '_zstd',
        'decimal', '_decimal', '_pydecimal',
        # GPL (readline) and ncurses: for interactive terminals, which the
        # add-on never has.
        'readline', 'curses', '_curses', '_curses_panel',
    ],
    noarchive=False,
)

a.binaries = _fix_libexpat(a.binaries)
a.binaries = [entry for entry in a.binaries if not os.path.basename(entry[0]).startswith('libmvec.so')]

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='bark3-addon',
    debug=False,
    bootloader_ignore_signals=False,
    # strip=False: stripping the native libraries saves little and breaks
    # macOS code signatures.
    strip=False,
    upx=False,
    console=True,  # stdio is the protocol channel
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='bark3-addon',
)
