"""The parts that need no model: features, unpacking, conversion, tokens, windows, phrases.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import lzma
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bark3_addon import convert, frontend, tokens, unpack, windows  # noqa: E402
from bark3_addon.decoder import Token  # noqa: E402
from bark3_addon.phrases import words_to_segments  # noqa: E402


# --- features --------------------------------------------------------------
def test_mel_filters_are_librosas():
    filters = frontend.mel_filters()
    assert filters.shape == (128, 257) and filters.dtype == np.float32
    assert (filters >= 0).all()
    assert filters[:, 0].max() == 0 and filters[:, 1:-1].max(axis=0).min() > 0
    # Values from librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm='slaney'),
    # which transformers' feature extractor uses.
    assert float(filters.sum()) == pytest.approx(4.090487480163574, rel=1e-6)
    for (band, bin_), value in {(0, 1): 0.02837754227221012, (64, 54): 0.011216705664992332,
                                (64, 55): 0.018818283453583717, (127, 244): 4.7641340643167496e-05,
                                (127, 245): 0.0009394199587404728}.items():
        assert filters[band, bin_] == pytest.approx(value, rel=1e-5)


def test_features_are_normalised_per_band():
    rng = np.random.default_rng(1)
    wave = (0.1 * rng.standard_normal(16000 * 3)).astype(np.float32)
    feats = frontend.features(wave)
    assert feats.shape == (301, 128)
    valid = frontend.frame_count(len(wave))
    assert np.allclose(feats[:valid].mean(axis=0), 0, atol=1e-4)
    assert np.allclose(feats[:valid].std(axis=0, ddof=1), 1, atol=1e-3)
    assert (feats[valid:] == 0).all()


def test_features_of_silence_stay_finite():
    assert np.isfinite(frontend.features(np.zeros(1600, np.float32))).all()


# --- unpacking: a .bark file written here, with each tensor kind -----------
def _ans_encode(symbols: np.ndarray, frequencies: np.ndarray) -> bytes:
    import constriction
    coder = constriction.stream.stack.AnsCoder()
    model = constriction.stream.model.Categorical(frequencies / frequencies.sum(), perfect=False)
    coder.encode_reverse(symbols.astype(np.int32), model)
    return coder.get_compressed().astype(np.uint32).tobytes()


def _write_bark(path: Path, rng) -> dict:
    """A bark-levels-v3 file holding one tensor of each kind; returns the expected values."""
    index, blobs, expected = [], [], {}

    # levels2: 6 rows x 10 columns, K=2, two column classes.
    rows, cols, k = 6, 10, 2
    levels = np.sort(rng.uniform(0.1, 1, (rows, k)), axis=1).astype(np.float16)
    codes = rng.integers(0, 2 * k + 1, (rows, cols))
    classes = np.array([c % 2 for c in range(cols)], np.uint8)
    bits = np.unpackbits(classes[:, None], axis=1)[:, 4:].reshape(-1)
    meta = np.packbits(bits).tobytes()
    counts = np.ones((16, 2 * k + 1), np.uint32)
    for cls in range(2):
        counts[cls] += np.bincount(codes[:, classes == cls].reshape(-1), minlength=2 * k + 1).astype(np.uint32)
    # One coder for all classes, decoded class by class, column-major: encode in that order.
    stream = []
    probabilities = []
    for cls in range(16):
        columns = np.nonzero(classes == cls)[0]
        stream.append(codes[:, columns].T.reshape(-1))
        probabilities += [counts[cls]] * (rows * len(columns))
    import constriction
    coder = constriction.stream.stack.AnsCoder()
    for symbols, cls in reversed([(s, c) for c, s in enumerate(stream) if len(s)]):
        model = constriction.stream.model.Categorical(counts[cls] / counts[cls].sum(), perfect=False)
        coder.encode_reverse(symbols.astype(np.int32), model)
    words = coder.get_compressed().astype(np.uint32).tobytes()
    lev = lzma.compress(levels.tobytes())
    meta_blob = meta + counts.tobytes()
    index.append({'n': 'encoder.a.weight', 'k': 'levels2', 'shape': [rows, cols], 'K': k,
                  'lev': len(lev), 'zl': 1, 'meta': len(meta_blob), 'b': len(words)})
    blobs += [lev, meta_blob, words]
    table = np.concatenate([np.zeros((rows, 1), np.float32), levels.astype(np.float32)], axis=1)
    mag = np.where(codes > k, codes - k, codes)
    expected['encoder.a.weight'] = np.where(codes > k, -1, 1) * np.take_along_axis(table, mag, 1)

    # ansint6: 4 rows x 5.
    q = rng.integers(0, 63, (4, 5))
    scale = rng.uniform(0.01, 0.1, 4).astype(np.float16)
    freqs = (np.bincount(q.reshape(-1), minlength=63) + 1).astype(np.uint32)
    words = _ans_encode(q.reshape(-1), freqs.astype(np.float64))
    index.append({'n': 'decoder.b.weight', 'k': 'ansint6', 'shape': [4, 5],
                  's': len(scale.tobytes()), 'meta': len(freqs.tobytes()), 'b': len(words)})
    blobs += [scale.tobytes(), freqs.tobytes(), words]
    expected['decoder.b.weight'] = (q - 31) * scale.astype(np.float32)[:, None]

    # raw fp16, LZMA'd, and a BatchNorm counter, which is skipped.
    bias = rng.standard_normal(7).astype(np.float16)
    raw = lzma.compress(bias.tobytes())
    index.append({'n': 'joint.c.bias', 'k': 'raw', 'shape': [7], 'b': len(raw), 'zr': 1})
    counter = np.array(5, np.int64).tobytes()
    index.append({'n': 'encoder.norm.num_batches_tracked', 'k': 'raw', 'shape': [],
                  'dtype': 'torch.int64', 'b': len(counter)})
    blobs += [raw, counter]
    expected['joint.c.bias'] = bias.astype(np.float32)

    header = json.dumps({'format': 'bark-levels-v3', 'index': index, 'config': '{"x": 1}'}).encode()
    path.write_bytes(len(header).to_bytes(8, 'little') + header + b''.join(blobs))
    return expected


def test_unpacks_every_tensor_kind_exactly(tmp_path):
    expected = _write_bark(tmp_path / 'model.bark', np.random.default_rng(7))
    got = dict(unpack.tensors(tmp_path / 'model.bark'))
    assert set(got) == set(expected)
    for name, value in expected.items():
        assert got[name].dtype == np.float32
        np.testing.assert_array_equal(got[name], value.astype(np.float32), err_msg=name)
    assert unpack.config(tmp_path / 'model.bark') == {'x': 1}


def test_a_file_that_is_not_bark_is_refused(tmp_path):
    (tmp_path / 'model.bark').write_bytes(b'\x00' * 64)
    with pytest.raises(unpack.FormatError):
        list(unpack.tensors(tmp_path / 'model.bark'))


def test_reads_bf16_safetensors(tmp_path):
    values = np.array([[1.5, -2.0], [0.25, 3.0]], np.float32)
    bf16 = (values.view(np.uint32) >> 16).astype('<u2').tobytes()
    counter = np.array(3, np.int64).tobytes()
    header = json.dumps({'joint.w': {'dtype': 'BF16', 'shape': [2, 2], 'data_offsets': [0, 8]},
                         'x.num_batches_tracked': {'dtype': 'I64', 'shape': [], 'data_offsets': [8, 16]},
                         '__metadata__': {}}).encode()
    path = tmp_path / 'model.safetensors'
    path.write_bytes(len(header).to_bytes(8, 'little') + header + bf16 + counter)
    got = dict(unpack.tensors(path))
    assert list(got) == ['joint.w']
    np.testing.assert_array_equal(got['joint.w'], values)


# --- conversion ------------------------------------------------------------
def test_quantization_is_per_output_channel():
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((6, 4)).astype(np.float32)  # [out, in]
    quantized, scale = convert._quantize(weight)
    assert quantized.shape == (4, 6) and quantized.dtype == np.int8 and scale.shape == (6,)
    assert np.abs(quantized).max(axis=0).tolist() == [127] * 6
    assert np.allclose(quantized * scale, weight.T, atol=scale.max() / 2 + 1e-7)
    # A 1x1 convolution [out, in, 1] is the same matrix.
    assert np.array_equal(convert._matrix(weight[:, :, None]), weight.T)


def test_a_folder_without_its_stamp_is_not_ready(tmp_path):
    assert not convert.is_ready(tmp_path)
    (tmp_path / convert.READY_FILE).write_text('{"graph_sha256": "old"}')
    assert not convert.is_ready(tmp_path)
    (tmp_path / convert.READY_FILE).write_text(json.dumps({'graph_sha256': convert.layout()['graph_sha256']}))
    assert convert.is_ready(tmp_path)


# --- tokens ----------------------------------------------------------------
def test_pieces_join_into_words_with_times():
    pieces = ['', '▁Hel', 'lo', ',', '▁world', '.', '▁']
    found = [Token(1, 10, 2), Token(2, 12, 1), Token(3, 13, 0), Token(0, 14, 1),
             Token(4, 20, 4), Token(5, 24, 1), Token(6, 25, 1)]
    words = tokens.words(found, pieces, offset_sec=1.0)
    assert [w['word'] for w in words] == ['Hello,', 'world.']
    assert words[0]['start'] == pytest.approx(1.8) and words[0]['end'] == pytest.approx(1 + 14 * 0.08)
    assert words[1]['start'] == pytest.approx(2.6) and words[1]['end'] == pytest.approx(1 + 25 * 0.08)


def test_the_vocabulary_drops_control_tokens(tmp_path):
    path = tmp_path / 'tokenizer.json'
    path.write_text(json.dumps({'model': {'vocab': {'<unk>': 0, '▁a': 1, 'b': 2}},
                                'added_tokens': [{'id': 0, 'content': '<unk>'}]}))
    assert tokens.load_vocabulary(path) == ['', '▁a', 'b']


# --- windows and phrases ---------------------------------------------------
def test_windows_cover_the_audio_and_cut_in_silence():
    rate = frontend.SAMPLE_RATE
    rng = np.random.default_rng(0)
    audio = (0.3 * rng.standard_normal(rate * 70)).astype(np.float32)
    audio[int(26.5 * rate):int(26.8 * rate)] = 0  # a pause inside the first search stretch
    ranges = windows.windows(audio)
    assert ranges[0][0] == 0 and ranges[-1][1] == len(audio)
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert 26.5 * rate <= ranges[0][1] <= 26.8 * rate
    assert all(end - start <= 35 * rate for start, end in ranges)
    assert windows.windows(audio[:rate * 20]) == [(0, rate * 20)]


def test_rows_close_at_sentence_ends():
    words = [{'word': 'Hello.', 'start': 0.0, 'end': 0.5}, {'word': 'Next', 'start': 0.6, 'end': 0.9},
             {'word': 'one', 'start': 0.9, 'end': 1.2}]
    rows = words_to_segments(words, 2.0)
    assert [r['text'] for r in rows] == ['Hello.', 'Next one']
