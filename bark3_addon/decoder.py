"""Greedy TDT decoding: the prediction network and joint, in numpy.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

A token-and-duration transducer reads the encoder's frames (80 ms each) with
a pointer. At each step the joint scores every token plus blank, and every
duration in DURATIONS. The best token is emitted unless it is blank; the
pointer then moves on by the best duration. A blank never stands still, and
after MAX_SYMBOLS tokens on one frame the pointer moves on regardless, as
NeMo's greedy TDT decoding does.

The prediction network (an embedding, two LSTM layers and a projection) only
runs when a token is emitted. Both networks are small (640 wide), so numpy
runs them faster than a graph call would: about 2% of real time on one core.
Many BLAS threads make these small products slower, which is why
OPENBLAS_NUM_THREADS is 1 (see __main__.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLANK = 8192
VOCAB = 8193
DURATIONS = (0, 1, 2, 3, 4)
MAX_SYMBOLS = 10
HIDDEN = 640
LAYERS = 2


@dataclass
class Token:
    id: int
    frame: int
    duration: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Decoder:
    """Runs on the decoder.* and joint.* tensors (float32) of convert.numpy_tensors()."""

    def __init__(self, tensors: dict[str, np.ndarray]):
        self.embedding = tensors['decoder.embedding.weight']
        self.lstm = []
        for layer in range(LAYERS):
            # One product per layer: [input; hidden] against both weight matrices.
            weights = np.concatenate([tensors[f'decoder.lstm.weight_ih_l{layer}'],
                                      tensors[f'decoder.lstm.weight_hh_l{layer}']], axis=1)
            bias = tensors[f'decoder.lstm.bias_ih_l{layer}'] + tensors[f'decoder.lstm.bias_hh_l{layer}']
            self.lstm.append((np.ascontiguousarray(weights), bias))
        self.projection = tensors['decoder.decoder_projector.weight']
        self.projection_bias = tensors['decoder.decoder_projector.bias']
        self.head = np.ascontiguousarray(tensors['joint.head.weight'])
        self.head_bias = tensors['joint.head.bias']

    def _predict(self, token: int, hidden: np.ndarray, cell: np.ndarray):
        x = self.embedding[token]
        new_hidden, new_cell = np.empty_like(hidden), np.empty_like(cell)
        for layer, (weights, bias) in enumerate(self.lstm):
            gates = weights @ np.concatenate([x, hidden[layer]]) + bias
            i, f, g, o = np.split(gates, 4)
            new_cell[layer] = _sigmoid(f) * cell[layer] + _sigmoid(i) * np.tanh(g)
            new_hidden[layer] = _sigmoid(o) * np.tanh(new_cell[layer])
            x = new_hidden[layer]
        return self.projection @ x + self.projection_bias, new_hidden, new_cell

    def decode(self, encoded: np.ndarray, cancel=lambda: False) -> list[Token]:
        """Tokens of `encoded` [steps, 640], each with its frame and predicted duration."""
        hidden = np.zeros((LAYERS, HIDDEN), np.float32)
        cell = np.zeros((LAYERS, HIDDEN), np.float32)
        # The network starts from blank, which is also the start token.
        prediction, next_hidden, next_cell = self._predict(BLANK, hidden, cell)
        tokens: list[Token] = []
        frame, steps, symbols = 0, len(encoded), 0
        while frame < steps:
            logits = self.head @ np.maximum(encoded[frame] + prediction, 0.0) + self.head_bias
            token = int(logits[:VOCAB].argmax())
            duration = DURATIONS[int(logits[VOCAB:].argmax())]
            if token != BLANK:
                tokens.append(Token(token, frame, duration))
                hidden, cell = next_hidden, next_cell
                prediction, next_hidden, next_cell = self._predict(token, hidden, cell)
                symbols += 1
                if duration == 0 and symbols >= MAX_SYMBOLS:
                    duration = 1
            elif duration == 0:
                duration = 1
            if duration:
                symbols = 0
                if cancel():
                    break
            frame += duration
        return tokens
