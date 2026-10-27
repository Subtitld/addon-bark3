"""Read Bark 3.0's weights, one tensor at a time.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Bark 3.0, Plus and Clean come as one `model.bark` file: an 8-byte length, a
JSON header, then each tensor's data in the header's order. ConwayResearch's
`bark_loader.py` documents the format ("bark-levels-v2", and v3, which adds
LZMA to the small side blobs). This module decodes it the same way, so the
weights are exactly the ones they evaluated:

- `levels2`: encoder linear weights. Each row keeps K magnitudes (fp16). Every
  weight is a code: 0, +level or -level. The codes are entropy-coded with ANS,
  with one symbol model for each of 16 column classes.
- `ansint5`, `ansint6`: symmetric 5- or 6-bit integers, ANS-coded, with one
  fp16 scale per row.
- `raw`: fp16 (or int64/int32) as stored.

Bark 3.0 Pro is a plain safetensors file in fp16 or bf16. `tensors()` reads
both formats and yields float32 arrays with transformers' ParakeetForTDT names.

ANS decoding needs `constriction` (MIT OR Apache-2.0), a native extension. It
takes most of the ~25 s that unpacking Bark 3.0 takes, which is why the add-on
unpacks once and keeps the result (see convert.py).
"""

from __future__ import annotations

import json
import lzma
import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np

FORMATS = ('bark-levels-v2', 'bark-levels-v3')
_COLUMN_CLASSES = 16


class FormatError(Exception):
    """The file is not a model this module reads."""


def _unlzma(data: bytes, compressed) -> bytes:
    return lzma.decompress(data) if compressed else data


def _ans(words: np.ndarray, probabilities: np.ndarray, count: int, coder=None):
    import constriction
    if coder is None:
        coder = constriction.stream.stack.AnsCoder(words)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)
    return coder.decode(model, count), coder


def _levels(entry: dict, blob: memoryview) -> np.ndarray:
    import constriction
    shape = entry['shape']
    rows, cols, k = shape[0], int(np.prod(shape[1:])), entry['K']
    pos = 0
    levels = np.frombuffer(_unlzma(bytes(blob[pos:pos + entry['lev']]), entry.get('zl')),
                           np.float16).reshape(rows, k).astype(np.float32)
    pos += entry['lev']
    meta = bytes(blob[pos:pos + entry['meta']])
    pos += entry['meta']
    words = np.frombuffer(bytes(blob[pos:pos + entry['b']]), np.uint32).copy()
    # Each column's class is a 4-bit number, most significant bit first.
    class_bytes = (cols * 4 + 7) // 8
    bits = np.unpackbits(np.frombuffer(meta[:class_bytes], np.uint8))[:cols * 4].reshape(cols, 4)
    classes = bits @ np.array([8, 4, 2, 1], np.uint8)
    counts = np.frombuffer(meta[class_bytes:], np.uint32).reshape(_COLUMN_CLASSES, 2 * k + 1)
    coder = constriction.stream.stack.AnsCoder(words)
    codes = np.empty((rows, cols), np.uint8)
    for cls in range(_COLUMN_CLASSES):
        columns = np.nonzero(classes == cls)[0]
        frequencies = counts[cls].astype(np.float64)
        symbols, coder = _ans(words, frequencies / frequencies.sum(), rows * len(columns), coder)
        codes[:, columns] = symbols.reshape(len(columns), rows).T
    # Code 0 is zero, 1..K are +level[code-1], K+1..2K are -level[code-K-1].
    magnitude = np.where(codes > k, codes - k, codes).astype(np.intp)
    table = np.concatenate([np.zeros((rows, 1), np.float32), levels], axis=1)
    values = np.take_along_axis(table, magnitude, axis=1)
    return np.where(codes > k, -values, values).reshape(shape)


def _ansint(entry: dict, blob: memoryview, bits: int) -> np.ndarray:
    shape = entry['shape']
    pos = 0
    scale = np.frombuffer(_unlzma(bytes(blob[pos:pos + entry['s']]), entry.get('zs')),
                          np.float16).astype(np.float32)
    pos += entry['s']
    frequencies = np.frombuffer(bytes(blob[pos:pos + entry['meta']]), np.uint32).astype(np.float64)
    pos += entry['meta']
    words = np.frombuffer(bytes(blob[pos:pos + entry['b']]), np.uint32).copy()
    symbols, _ = _ans(words, frequencies / frequencies.sum(), int(np.prod(shape)))
    values = (symbols - (2 ** (bits - 1) - 1)).astype(np.float32)
    return (values.reshape(len(scale), -1) * scale[:, None]).reshape(shape)


def _raw(entry: dict, blob: memoryview) -> np.ndarray:
    dtype = {'torch.int64': np.int64, 'torch.int32': np.int32}.get(entry.get('dtype'), np.float16)
    data = _unlzma(bytes(blob[:entry['b']]), entry.get('zr'))
    return np.frombuffer(data, dtype).reshape(entry['shape'])


def _entry_size(entry: dict) -> int:
    kind = entry['k']
    if kind == 'levels2':
        return entry['lev'] + entry['meta'] + entry['b']
    if kind.startswith('ansint'):
        return entry['s'] + entry['meta'] + entry['b']
    if kind == 'raw':
        return entry['b']
    raise FormatError(f'{entry["n"]}: unknown tensor kind {kind!r}')


def bark_header(path: Path) -> tuple[dict, int]:
    """The JSON header of a .bark file, and where the tensor data starts."""
    with open(path, 'rb') as handle:
        size = int.from_bytes(handle.read(8), 'little')
        if not 0 < size < 1 << 26:
            raise FormatError(f'{path.name}: not a .bark file')
        header = json.loads(handle.read(size))
    if header.get('format') not in FORMATS:
        raise FormatError(f'{path.name}: format {header.get("format")!r}, this add-on reads {", ".join(FORMATS)}')
    return header, 8 + size


def _bark_tensors(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    header, start = bark_header(path)
    data = memoryview(np.memmap(path, np.uint8, mode='r'))
    pos = start
    for entry in header['index']:
        size = _entry_size(entry)
        blob = data[pos:pos + size]
        pos += size
        kind = entry['k']
        if entry['n'].endswith('num_batches_tracked'):
            continue  # a BatchNorm training counter, not a weight
        if kind == 'levels2':
            yield entry['n'], _levels(entry, blob)
        elif kind.startswith('ansint'):
            yield entry['n'], _ansint(entry, blob, int(kind[6:]))
        else:
            yield entry['n'], _raw(entry, blob).astype(np.float32)
    if pos != len(data):
        raise FormatError(f'{path.name}: {len(data) - pos} bytes after the last tensor')


_SAFETENSORS_TYPES = {'F32': np.float32, 'F16': np.float16, 'BF16': None}


def _safetensors(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    with open(path, 'rb') as handle:
        size = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(size))
    data = np.memmap(path, np.uint8, mode='r', offset=8 + size)
    for name, meta in header.items():
        if name == '__metadata__' or name.endswith('num_batches_tracked'):
            continue
        dtype = meta['dtype']
        if dtype not in _SAFETENSORS_TYPES:
            raise FormatError(f'{path.name}: {name} is {dtype}')
        begin, end = meta['data_offsets']
        raw = data[begin:end]
        if dtype == 'BF16':
            # bf16 is the top half of a float32.
            halves = raw.view('<u2').astype(np.uint32) << 16
            array = halves.view(np.float32)
        else:
            array = raw.view(np.dtype(_SAFETENSORS_TYPES[dtype]).newbyteorder('<')).astype(np.float32)
        yield name, array.reshape(meta['shape'])


def tensors(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """(name, float32 array) for every weight in a model.bark or model.safetensors."""
    path = Path(path)
    if path.suffix == '.safetensors':
        yield from _safetensors(path)
    else:
        yield from _bark_tensors(path)


def config(path: Path) -> dict:
    """The model configuration a .bark file carries (ParakeetTDTConfig as a dict)."""
    header, _ = bark_header(path)
    return json.loads(header['config'])
