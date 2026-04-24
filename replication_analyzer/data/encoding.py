"""
Signal encoding functions for BrdU/EdU replication signals.

This module provides multi-channel encoding strategies to transform
raw signals into rich feature representations for deep learning models.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from typing import Optional
import pywt


def encode_signal_multichannel_basic(signal: np.ndarray,
                                     smooth_sigma: float = 2,
                                     window: int = 50) -> np.ndarray:
    """
    Basic 6-channel signal encoding.

    Channels:
    0. Normalized signal
    1. Smoothed signal
    2. Gradient (1st derivative)
    3. Second derivative
    4. Local mean
    5. Local std

    Parameters
    ----------
    signal : np.ndarray
        Raw signal values (1D array)
    smooth_sigma : float
        Sigma for Gaussian smoothing
    window : int
        Window size for local statistics

    Returns
    -------
    np.ndarray
        Encoded signal with shape (n_segments, 6)
    """
    signal = np.array(signal, dtype=np.float32)
    n = len(signal)

    if n < 5:
        return np.zeros((n, 6), dtype=np.float32)

    # Normalize (z-score)
    mean = np.mean(signal)
    std = np.std(signal) + 1e-8
    norm_signal = (signal - mean) / std

    # Smoothed signal
    smooth_signal = gaussian_filter1d(norm_signal, sigma=min(smooth_sigma, n / 10))

    # Gradient and 2nd derivative
    grad = np.gradient(smooth_signal)
    grad2 = np.gradient(grad)

    # Adjust window if shorter than read
    win = min(window, max(3, n // 2))
    kernel = np.ones(win) / win

    # Rolling mean and std
    local_mean = np.convolve(norm_signal, kernel, mode='same')
    local_std = np.sqrt(np.convolve((norm_signal - local_mean)**2, kernel, mode='same'))

    # Combine into multi-channel feature map
    X = np.stack([
        norm_signal,
        smooth_signal,
        grad,
        grad2,
        local_mean,
        local_std
    ], axis=1)

    return X.astype(np.float32)


def encode_signal_multichannel_enhanced(signal: np.ndarray,
                                        smooth_sigma: float = 2,
                                        window: int = 50) -> np.ndarray:
    """
    Enhanced 9-channel signal encoding (used in EXPERT MODEL).

    Channels:
    0. Normalized signal
    1. Smoothed signal
    2. Gradient (1st derivative)
    3. Second derivative
    4. Local mean
    5. Local std
    6. Z-score (deviation from local mean)
    7. Cumulative sum (trend detection)
    8. Signal envelope

    Parameters
    ----------
    signal : np.ndarray
        Raw signal values (1D array)
    smooth_sigma : float
        Sigma for Gaussian smoothing
    window : int
        Window size for local statistics

    Returns
    -------
    np.ndarray
        Encoded signal with shape (n_segments, 9)
    """
    signal = np.array(signal, dtype=np.float32)
    n = len(signal)

    if n < 5:
        return np.zeros((n, 9), dtype=np.float32)

    # Normalize
    mean = np.mean(signal)
    std = np.std(signal) + 1e-8
    norm_signal = (signal - mean) / std

    # Smooth
    smooth_signal = gaussian_filter1d(norm_signal, sigma=min(smooth_sigma, n / 10))

    # Derivatives
    grad = np.gradient(smooth_signal)
    grad2 = np.gradient(grad)

    # Local statistics
    win = min(window, max(3, n // 2))
    kernel = np.ones(win) / win
    local_mean = np.convolve(norm_signal, kernel, mode='same')
    local_std = np.sqrt(np.convolve((norm_signal - local_mean)**2, kernel, mode='same'))

    # Z-score (standardized deviation from local mean)
    z_score = (norm_signal - local_mean) / (local_std + 1e-8)

    # Cumulative sum (trend detection)
    cumsum = np.cumsum(norm_signal - norm_signal.mean())
    cumsum = (cumsum - cumsum.mean()) / (np.std(cumsum) + 1e-8)

    # Signal envelope (Hilbert-like approximation using moving max/min)
    env_max = maximum_filter1d(norm_signal, size=min(20, n//3), mode='nearest')
    env_min = minimum_filter1d(norm_signal, size=min(20, n//3), mode='nearest')
    envelope = (env_max - env_min) / 2

    # Stack channels
    X = np.stack([
        norm_signal,      # 0: Normalized signal
        smooth_signal,    # 1: Smooth signal
        grad,            # 2: 1st derivative
        grad2,           # 3: 2nd derivative
        local_mean,      # 4: Local mean
        local_std,       # 5: Local std
        z_score,         # 6: Z-score
        cumsum,          # 7: Cumulative trend
        envelope         # 8: Envelope
    ], axis=1)

    return X.astype(np.float32)


def encode_signal_wavelet_enhanced(signal: np.ndarray,
                                    wavelet: str = 'db4',
                                    level: int = 2,
                                    window: int = 50) -> np.ndarray:
    """
    Wavelet-based 9-channel signal encoding for improved edge detection.

    Uses wavelet decomposition instead of Gaussian smoothing to better
    detect sharp transitions (forks/origins).

    Channels:
    0. Normalized signal
    1. Wavelet approximation (low-frequency trend)
    2. Wavelet detail level 1 (high-frequency edges)
    3. Wavelet detail level 2 (medium-frequency features)
    4. Local mean
    5. Local std
    6. Z-score (deviation from local mean)
    7. Cumulative sum (trend detection)
    8. Signal envelope

    Parameters
    ----------
    signal : np.ndarray
        Raw signal values (1D array)
    wavelet : str
        Wavelet type (default: 'db4' - Daubechies 4)
    level : int
        Decomposition level (default: 2)
    window : int
        Window size for local statistics

    Returns
    -------
    np.ndarray
        Encoded signal with shape (n_segments, 9)
    """
    signal = np.array(signal, dtype=np.float32)
    n = len(signal)

    if n < 5:
        return np.zeros((n, 9), dtype=np.float32)

    # Normalize
    mean = np.mean(signal)
    std = np.std(signal) + 1e-8
    norm_signal = (signal - mean) / std

    # Wavelet decomposition (replaces Gaussian smoothing)
    try:
        coeffs = pywt.wavedec(norm_signal, wavelet, level=level)

        # Approximation (low-frequency trend)
        approx = coeffs[0]
        approx_upsampled = np.repeat(approx, len(norm_signal) // len(approx) + 1)[:len(norm_signal)]

        # Detail coefficients (high-frequency edges)
        detail1 = coeffs[1] if len(coeffs) > 1 else np.zeros_like(coeffs[0])
        detail1_upsampled = np.repeat(detail1, len(norm_signal) // len(detail1) + 1)[:len(norm_signal)]

        detail2 = coeffs[2] if len(coeffs) > 2 else np.zeros_like(coeffs[0])
        detail2_upsampled = np.repeat(detail2, len(norm_signal) // len(detail2) + 1)[:len(norm_signal)]

    except Exception:
        # Fallback to Gaussian if wavelet fails
        approx_upsampled = gaussian_filter1d(norm_signal, sigma=2)
        detail1_upsampled = np.gradient(approx_upsampled)
        detail2_upsampled = np.gradient(detail1_upsampled)

    # Local statistics
    win = min(window, max(3, n // 2))
    kernel = np.ones(win) / win
    local_mean = np.convolve(norm_signal, kernel, mode='same')
    local_std = np.sqrt(np.convolve((norm_signal - local_mean)**2, kernel, mode='same'))

    # Z-score
    z_score = (norm_signal - local_mean) / (local_std + 1e-8)

    # Cumulative sum
    cumsum = np.cumsum(norm_signal - norm_signal.mean())
    cumsum = (cumsum - cumsum.mean()) / (np.std(cumsum) + 1e-8)

    # Envelope
    env_max = maximum_filter1d(norm_signal, size=min(20, n//3), mode='nearest')
    env_min = minimum_filter1d(norm_signal, size=min(20, n//3), mode='nearest')
    envelope = (env_max - env_min) / 2

    # Stack channels
    X = np.stack([
        norm_signal,         # 0: Normalized signal
        approx_upsampled,    # 1: Wavelet approximation (trend)
        detail1_upsampled,   # 2: Detail 1 (edges)
        detail2_upsampled,   # 3: Detail 2 (features)
        local_mean,          # 4: Local mean
        local_std,           # 5: Local std
        z_score,             # 6: Z-score
        cumsum,              # 7: Cumulative trend
        envelope             # 8: Envelope
    ], axis=1)

    return X.astype(np.float32)


# Alias for backward compatibility and convenience
encode_signal_enhanced = encode_signal_multichannel_enhanced
encode_signal_basic = encode_signal_multichannel_basic
encode_signal_wavelet = encode_signal_wavelet_enhanced
