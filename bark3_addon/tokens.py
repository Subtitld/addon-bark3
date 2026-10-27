"""Tokens to words with times.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The vocabulary is parakeet-tdt-0.6b-v3's SentencePiece one, read from its
tokenizer.json (downloaded with the model, see models.py). A piece that starts
with "▁" starts a word; the others, punctuation included, continue the one
before. Control tokens (<unk>, <pad>, <|...|>) are dropped, as transformers'
decode with skip_special_tokens drops them.
"""

from __future__ import annotations

import json
from pathlib import Path

from bark3_addon.decoder import Token

FRAME_SEC = 0.08  # the encoder's frame: 8 feature frames of 10 ms
_WORD_START = '▁'


def load_vocabulary(path: Path) -> list[str]:
    """Piece text by id, with control tokens as ''."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    vocab = data['model']['vocab']
    pieces = [''] * (max(vocab.values()) + 1)
    for piece, index in vocab.items():
        pieces[index] = piece
    for added in data.get('added_tokens', []):
        if added['id'] < len(pieces):
            pieces[added['id']] = ''
    return pieces


def words(tokens: list[Token], pieces: list[str], offset_sec: float = 0.0) -> list[dict]:
    """[{word, start, end}] in seconds, `offset_sec` added to every time."""
    out: list[dict] = []
    for token in tokens:
        piece = pieces[token.id] if token.id < len(pieces) else ''
        if not piece:
            continue
        start = token.frame * FRAME_SEC + offset_sec
        end = (token.frame + max(1, token.duration)) * FRAME_SEC + offset_sec
        if piece.startswith(_WORD_START) or not out:
            text = piece.lstrip(_WORD_START)
            if not text:
                continue
            out.append({'word': text, 'start': start, 'end': end})
        else:
            out[-1]['word'] += piece
            out[-1]['end'] = max(out[-1]['end'], end)
    return out
