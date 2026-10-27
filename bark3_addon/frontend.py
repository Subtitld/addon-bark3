"""Audio to the encoder's input: 128 log-mel features every 10 ms.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The same computation as transformers' ParakeetFeatureExtractor (NeMo's
preprocessor for parakeet-tdt-0.6b-v3), in numpy: pre-emphasis 0.97, a 512-point
STFT of 25 ms symmetric Hann windows every 10 ms, centred with zero padding,
power spectrum, 128 Slaney mel bands, log, then each band normalised to zero
mean and unit variance over the clip.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 160
WINDOW = 400
N_MELS = 128
PREEMPHASIS = 0.97
_LOG_GUARD = 2.0 ** -24
_EPSILON = 1e-5


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    hz = np.asarray(hz, np.float64)
    mel = hz / (200.0 / 3)
    log_region = hz >= 1000.0
    mel[log_region] = 15.0 + np.log(hz[log_region] / 1000.0) / (np.log(6.4) / 27.0)
    return mel


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, np.float64)
    hz = mel * (200.0 / 3)
    log_region = mel >= 15.0
    hz[log_region] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mel[log_region] - 15.0))
    return hz


@lru_cache(maxsize=1)
def mel_filters() -> np.ndarray:
    """[128, 257] Slaney-normalised triangles over 0..8 kHz, as librosa.filters.mel makes them."""
    bins = np.linspace(0, SAMPLE_RATE / 2, 1 + N_FFT // 2)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(np.array([0.0]))[0],
                                   _hz_to_mel(np.array([SAMPLE_RATE / 2]))[0], N_MELS + 2))
    widths = np.diff(edges)
    ramps = edges[:, None] - bins[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    filters = np.maximum(0, np.minimum(lower, upper))
    filters *= (2.0 / (edges[2:] - edges[:-2]))[:, None]
    return filters.astype(np.float32)


@lru_cache(maxsize=1)
def _window() -> np.ndarray:
    window = np.zeros(N_FFT, np.float32)
    start = (N_FFT - WINDOW) // 2
    window[start:start + WINDOW] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WINDOW) / (WINDOW - 1))
    return window


def frame_count(samples: int) -> int:
    """How many feature frames count for `samples` samples (the rest is padding)."""
    return (samples + N_FFT // 2 * 2 - N_FFT) // HOP


def features(wave: np.ndarray) -> np.ndarray:
    """[frames, 128] float32 features of mono 16 kHz float samples."""
    wave = np.asarray(wave, np.float32)
    emphasised = np.concatenate([wave[:1], wave[1:] - PREEMPHASIS * wave[:-1]])
    padded = np.pad(emphasised, N_FFT // 2)
    count = 1 + (len(padded) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        padded, (count, N_FFT), (padded.strides[0] * HOP, padded.strides[0]), writeable=False)
    power = np.abs(np.fft.rfft(frames * _window(), axis=1)) ** 2
    mel = np.log(power.astype(np.float32) @ mel_filters().T + _LOG_GUARD)
    valid = max(1, min(frame_count(len(wave)), count))
    head = mel[:valid]
    mean = head.mean(axis=0)
    std = np.sqrt(((head - mean) ** 2).sum(axis=0) / max(1, valid - 1))
    out = (mel - mean) / (std + _EPSILON)
    out[valid:] = 0.0
    return out.astype(np.float32)
