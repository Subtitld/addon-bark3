"""Turn a downloaded model into the files the engine runs: once per model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The add-on carries the encoder's graph without weights (graph/encoder.onnx)
and the layout of the file its weights are read from (graph/layout.json),
both made by tools/build_graph.py. Here each tensor of the downloaded model is
decoded (unpack.py) and written into `weights.bin` at the place the layout
gives it, in the form the layout asks for:

- `copy`: as is (biases, norms, convolutions; the decoder and joint, which
  numpy runs in float32);
- `transpose`: a matrix stored transposed, as the graph's MatMul takes it;
- `qweight`, `qscale`, `qzero`: the transposed matrix quantized to int8 for
  ONNX Runtime's MatMulInteger, symmetric, one scale per output channel.

The result is a folder holding weights.bin, a copy of encoder.onnx (ONNX
Runtime looks for external data next to the graph) and `ready.json`, written
last. Without `ready.json` the folder is unfinished and is rebuilt. Peak memory
is one decoded tensor, not the model.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bark3_addon import unpack

GRAPH_DIR = Path(__file__).resolve().parent / 'graph'
WEIGHTS_FILE = 'weights.bin'
GRAPH_FILE = 'encoder.onnx'
READY_FILE = 'ready.json'


class ConvertError(Exception):
    """The model does not fit the graph."""


class Cancelled(Exception):
    """The conversion was cancelled."""


def layout() -> dict:
    return json.loads((GRAPH_DIR / 'layout.json').read_text(encoding='utf-8'))


def is_ready(folder: Path) -> bool:
    """True when `folder` was built, completely, for this add-on's graph."""
    try:
        stamp = json.loads((Path(folder) / READY_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return stamp.get('graph_sha256') == layout()['graph_sha256']


def _matrix(array: np.ndarray) -> np.ndarray:
    """A linear weight [out, in] or a 1x1 convolution [out, in, 1], transposed to [in, out]."""
    return array.reshape(array.shape[0], -1).T


def _quantize(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = _matrix(array)
    scale = np.abs(matrix).max(axis=0) / 127.0
    scale[scale == 0] = 1.0
    quantized = np.clip(np.rint(matrix / scale), -127, 127).astype(np.int8)
    return quantized, scale.astype(np.float32)


def _forms(array: np.ndarray, recipes: set[str]) -> dict[str, np.ndarray]:
    forms = {}
    if 'copy' in recipes:
        forms['copy'] = array
    if 'transpose' in recipes:
        forms['transpose'] = _matrix(array)
    if recipes & {'qweight', 'qscale', 'qzero'}:
        forms['qweight'], forms['qscale'] = _quantize(array)
        forms['qzero'] = np.zeros(forms['qscale'].shape, np.int8)
    return forms


def build(model_path: Path, folder: Path, progress: Callable[[int, int], None],
          cancel: Callable[[], bool] = lambda: False) -> Path:
    """Write `folder` from the model at `model_path`. `progress(done, total)` counts tensors."""
    spec = layout()
    folder = Path(folder)
    partial = folder.with_name(folder.name + '.part')
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for entry in spec['tensors']:
        by_source[entry['source']].append(entry)
    weights_path = partial / WEIGHTS_FILE
    with open(weights_path, 'wb') as handle:
        handle.truncate(spec['weights_bytes'])
    weights = np.memmap(weights_path, np.uint8, mode='r+')
    done, total, written = 0, len(by_source), set()
    try:
        for name, array in unpack.tensors(model_path):
            if cancel():
                raise Cancelled()
            entries = by_source.get(name)
            if not entries:
                continue
            forms = _forms(array, {e['recipe'] for e in entries})
            for entry in entries:
                value = np.ascontiguousarray(forms[entry['recipe']], dtype=entry['dtype'])
                if list(value.shape) != entry['shape']:
                    raise ConvertError(f'{name}: shape {list(value.shape)}, the graph wants {entry["shape"]}')
                weights[entry['offset']:entry['offset'] + entry['nbytes']] = value.reshape(-1).view(np.uint8)
            written.add(name)
            done += 1
            progress(done, total)
        weights.flush()
    finally:
        del weights
    absent = sorted(set(by_source) - written)
    if absent:
        raise ConvertError(f'{Path(model_path).name} lacks {len(absent)} tensors, such as {absent[0]}')
    shutil.copyfile(GRAPH_DIR / GRAPH_FILE, partial / GRAPH_FILE)
    (partial / READY_FILE).write_text(json.dumps({'graph_sha256': spec['graph_sha256']}) + '\n',
                                      encoding='utf-8')
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(partial, folder)
    return folder


def numpy_tensors(folder: Path) -> dict[str, np.ndarray]:
    """The decoder's and joint's float32 tensors, memory-mapped from weights.bin."""
    data = np.memmap(Path(folder) / WEIGHTS_FILE, np.uint8, mode='r')
    out = {}
    for entry in layout()['tensors']:
        if entry['name'].startswith(('decoder.', 'joint.')):
            raw = data[entry['offset']:entry['offset'] + entry['nbytes']]
            out[entry['name']] = raw.view(entry['dtype']).reshape(entry['shape'])
    return out
