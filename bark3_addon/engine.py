"""One Bark 3.0 model, ready to transcribe.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The encoder (FastConformer, int8) runs in ONNX Runtime from the folder
convert.py built; the features, the prediction network and the joint run in
numpy. Each call takes one window of audio (see windows.py); attention covers
the whole window, so memory grows with the square of its length.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from bark3_addon import convert, frontend, tokens
from bark3_addon.decoder import Decoder

log = logging.getLogger('bark3_addon.engine')


class Engine:
    """Calls are serialised; one transcription at a time per engine."""

    def __init__(self, folder: Path, vocabulary: Path, threads: int, pad_sec: float):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(str(Path(folder) / convert.GRAPH_FILE), options,
                                             providers=['CPUExecutionProvider'])
        self._decoder = Decoder(convert.numpy_tensors(folder))
        self._pieces = tokens.load_vocabulary(vocabulary)
        self._pad = np.zeros(int(round(pad_sec * frontend.SAMPLE_RATE)), np.float32)
        self._pad_sec = len(self._pad) / frontend.SAMPLE_RATE
        self._lock = threading.Lock()

    def transcribe(self, samples: np.ndarray, offset_sec: float = 0.0,
                   cancel=lambda: False) -> list[dict]:
        """Words of 16 kHz mono float samples, with times from `offset_sec`.

        The model was evaluated with a little silence before the speech (0.25 s
        for Bark 3.0, Plus and Pro, none for Clean); it is added here and its
        length taken off the times.
        """
        wave = np.concatenate([self._pad, np.asarray(samples, np.float32)])
        features = frontend.features(wave)[None]
        with self._lock:
            encoded = self._session.run(None, {'features': features})[0][0]
            found = self._decoder.decode(encoded, cancel)
        result = tokens.words(found, self._pieces, offset_sec - self._pad_sec)
        log.debug('window at %.1fs: %d samples, features %s (|max| %.2f), encoded %s (|max| %.2f), '
                  '%d tokens, %d words', offset_sec, len(wave), features.shape,
                  float(np.abs(features).max()), encoded.shape, float(np.abs(encoded).max()),
                  len(found), len(result))
        for word in result:
            word['start'] = max(offset_sec, word['start'])
            word['end'] = max(word['start'], word['end'])
        return result
