"""Cut long audio into windows the encoder takes one at a time.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The encoder's attention spans its whole input, so its memory grows with the
square of the length: a one-hour file in one call would need tens of
gigabytes. Windows of about WINDOW_SEC keep it near 1 GB, report progress and
let a cancel take effect. Each window is cut at the quietest moment near its
end, where no word is likely to be split. The rule is addon-whistle's, in numpy.
"""

from __future__ import annotations

import numpy as np

from bark3_addon.frontend import SAMPLE_RATE

WINDOW_SEC = 30.0
# The cut is searched for over this last stretch of each window.
SEARCH_SEC = 8.0
# A tail shorter than this joins the window before it.
MIN_TAIL_SEC = 5.0
# Loudness is measured over frames this long.
_FRAME = SAMPLE_RATE // 10


def windows(samples: np.ndarray, window_sec: float = WINDOW_SEC, search_sec: float = SEARCH_SEC,
            min_tail_sec: float = MIN_TAIL_SEC) -> list[tuple[int, int]]:
    """(start, end) sample ranges that cover `samples` end to end, in order."""
    total = len(samples)
    window = int(window_sec * SAMPLE_RATE)
    search = int(search_sec * SAMPLE_RATE)
    min_tail = int(min_tail_sec * SAMPLE_RATE)
    ranges: list[tuple[int, int]] = []
    start = 0
    while total - start > window + min_tail:
        first = start + window - search
        count = search // _FRAME
        frames = np.asarray(samples[first:first + count * _FRAME], np.float32).reshape(count, _FRAME)
        quietest = int(np.argmin(np.einsum('ij,ij->i', frames, frames)))
        cut = first + quietest * _FRAME + _FRAME // 2  # the middle of the quietest frame
        ranges.append((start, cut))
        start = cut
    if start < total:
        ranges.append((start, total))
    return ranges
