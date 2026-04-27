"""
Encoding functions for RECTANGULAR BLOCK signal representation.

CRITICAL DIFFERENCE from standard encoding.py:
- Standard: Each bin treated as single point value
- Rectangular: Each bin expanded to its full genomic width before encoding

This preserves the step-function nature of binned BrdU/EdU signal data.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
import pandas as pd
import pywt


def expand_bins_to_rectangular_signal(bins_df: pd.DataFrame, target_length: int = 10000) -> np.ndarray:
    """
    Expand binned signal to rectangular step-function representation.

    CRITICAL: This is the key difference from point-value encoding.
    Each bin's constant value is repeated for its full genomic extent.

    Parameters
    ----------
    bins_df : pd.DataFrame
        DataFrame with columns: 'start', 'end', 'signal'
    target_length : int
        Target length for the expanded signal (will be resampled to this)

    Returns
    -------
    np.ndarray
        Expanded signal as rectangular step-function

    Example
    -------
    Input bins:
        start   end   signal
        1000   1100    0.5   (100bp wide)
        1100   1300    0.8   (200bp wide)

    Point-value approach (OLD - WRONG):
        signal = [0.5, 0.8]  (2 values)

    Rectangular block approach (NEW - CORRECT):
        signal = [0.5]*100 + [0.8]*200  (300 values)
        Then resampled to target_length
    """
    if len(bins_df) == 0:
        return np.zeros(target_length, dtype=np.float32)

    # Build rectangular step-function signal using vectorized bin mapping.
    # This avoids expanding to base-pair resolution (which caused OOM for long reads).
    widths = (bins_df['end'].values - bins_df['start'].values).astype(np.float64)
    widths = np.maximum(widths, 0)
    signals = bins_df['signal'].values.astype(np.float32)

    total_width = widths.sum()
    if total_width <= 0 or len(signals) == 0:
        return np.zeros(target_length, dtype=np.float32)

    # Map each target index to the bin it falls in, weighted by genomic width.
    # searchsorted on cumulative widths gives the step-function directly.
    cum_widths = np.cumsum(widths)
    target_positions = np.arange(target_length, dtype=np.float64) * total_width / target_length
    bin_indices = np.searchsorted(cum_widths, target_positions, side='right')
    bin_indices = np.clip(bin_indices, 0, len(signals) - 1)

    return signals[bin_indices].astype(np.float32)


def encode_signal_rectangular_gaussian(signal: np.ndarray,
                                       smooth_sigma: float = 2,
                                       window: int = 50) -> np.ndarray:
    """
    Multi-channel encoding for RECTANGULAR BLOCK signal.

    CRITICAL: Input signal is already expanded rectangular step-function.
    This is applied AFTER expand_bins_to_rectangular_signal().

    Creates 6 channels:
    1. Normalized signal (preserves steps)
    2. Smoothed signal (Gaussian)
    3. Gradient
    4. 2nd derivative
    5. Local mean
    6. Local std

    Parameters
    ----------
    signal : np.ndarray
        EXPANDED rectangular step-function signal
    smooth_sigma : float
        Gaussian smoothing sigma
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

    # Smoothed signal (Gaussian preserves general shape but smooths steps)
    smooth_signal = gaussian_filter1d(norm_signal, sigma=min(smooth_sigma, n / 10))

    # Gradient and 2nd derivative (will show sharp transitions at step boundaries)
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


def encode_signal_rectangular_wavelet(signal: np.ndarray,
                                      smooth_sigma: float = 2,
                                      window: int = 50,
                                      wavelet: str = 'db4',
                                      level: int = 2,
                                      extra_channels: bool = False) -> np.ndarray:
    """
    Multi-channel encoding using WAVELET decomposition for RECTANGULAR BLOCK signal.

    CRITICAL: Input signal is already expanded rectangular step-function.
    Wavelet decomposition will capture sharp transitions at block boundaries.

    Creates 11 channels:
    1. Normalized signal
    2. Wavelet approximation (low-freq)
    3. Wavelet detail 1 (high-freq - captures step edges)
    4. Wavelet detail 2 (high-freq - captures step edges)
    5. Local mean
    6. Local std
    7. Z-score
    8. Cumulative sum
    9. Envelope
    10. Local slope  (gradient of smoothed signal — positive=rising, negative=falling)
    11. Read-level mean BrdU  (constant per read — distinguishes null-BrdU reads)

    Parameters
    ----------
    signal : np.ndarray
        EXPANDED rectangular step-function signal
    smooth_sigma : float
        Fallback Gaussian smoothing sigma
    window : int
        Window size for local statistics
    wavelet : str
        Wavelet type (default: 'db4' - good for sharp edges)
    level : int
        Decomposition level

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

    # Wavelet decomposition (CRITICAL: captures sharp step transitions)
    try:
        coeffs = pywt.wavedec(norm_signal, wavelet, level=level)

        # Approximation (low-frequency trend)
        approx = coeffs[0]
        approx_upsampled = np.repeat(approx, len(norm_signal) // len(approx) + 1)[:len(norm_signal)]

        # Detail coefficients (high-frequency edges - will highlight step boundaries!)
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

    # Z-score (local normalization)
    z_score = (norm_signal - local_mean) / (local_std + 1e-8)

    # Cumulative sum (trend detection)
    cumsum = np.cumsum(norm_signal - norm_signal.mean())
    cumsum = (cumsum - cumsum.mean()) / (np.std(cumsum) + 1e-8)

    # Envelope (signal range)
    env_max = maximum_filter1d(norm_signal, size=min(20, n//3), mode='nearest')
    env_min = minimum_filter1d(norm_signal, size=min(20, n//3), mode='nearest')
    envelope = (env_max - env_min) / 2

    channels = [
        norm_signal,
        approx_upsampled,
        detail1_upsampled,
        detail2_upsampled,
        local_mean,
        local_std,
        z_score,
        cumsum,
        envelope,
    ]

    if extra_channels:
        # Ch10: local slope — gradient of heavily smoothed signal
        # Positive = rising left→right (RF), negative = falling (LF), ~0 = ORI/background
        slope_sigma = max(smooth_sigma * 3, 5)
        smoothed_for_slope = gaussian_filter1d(signal, sigma=slope_sigma)
        local_slope_raw = np.gradient(smoothed_for_slope)
        local_slope = local_slope_raw / (np.std(local_slope_raw) + 1e-8)

        # Ch11: read-level mean BrdU — constant per read, ~0 on null-BrdU reads
        read_mean_brdu = np.full(n, mean, dtype=np.float32)

        channels += [local_slope, read_mean_brdu]

    # Combine into multi-channel feature map
    X = np.stack(channels, axis=1)

    return X.astype(np.float32)
