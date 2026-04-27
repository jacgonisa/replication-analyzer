"""Shared train/predict encoding path for the CODEX pipeline."""

from __future__ import annotations

import pandas as pd

from replication_analyzer.data.encoding import (
    encode_signal_basic,
    encode_signal_enhanced,
    encode_signal_wavelet,
)
from replication_analyzer.data.encoding_rectangular import (
    expand_bins_to_rectangular_signal,
    encode_signal_rectangular_gaussian,
    encode_signal_rectangular_wavelet,
)


def encode_read_dataframe(read_df: pd.DataFrame, preprocessing_config: dict):
    """
    Encode one read using a single config-driven path shared by train and predict.

    The output length always matches `len(read_df)` so labels and predictions stay aligned
    to the original genomic windows even when the signal is rectangularized internally.
    """
    signal_mode = preprocessing_config.get("signal_mode", "wavelet")
    use_rectangular = preprocessing_config.get("use_rectangular_blocks", True)
    smooth_sigma = preprocessing_config.get("smooth_sigma", 2)
    local_window = preprocessing_config.get("local_window", 50)
    wavelet = preprocessing_config.get("wavelet", "db4")
    wavelet_level = preprocessing_config.get("wavelet_level", 2)

    if use_rectangular:
        bins_df = read_df[["start", "end", "signal"]].copy()
        rectangular_signal = expand_bins_to_rectangular_signal(
            bins_df=bins_df,
            target_length=len(read_df),
        )
        if signal_mode == "gaussian":
            return encode_signal_rectangular_gaussian(
                rectangular_signal,
                smooth_sigma=smooth_sigma,
                window=local_window,
            )
        return encode_signal_rectangular_wavelet(
            rectangular_signal,
            smooth_sigma=smooth_sigma,
            window=local_window,
            wavelet=wavelet,
            level=wavelet_level,
            extra_channels=preprocessing_config.get("extra_channels", False),
        )

    raw_signal = read_df["signal"].to_numpy()
    if signal_mode == "basic":
        return encode_signal_basic(
            raw_signal,
            smooth_sigma=smooth_sigma,
            window=local_window,
        )
    if signal_mode == "enhanced":
        return encode_signal_enhanced(
            raw_signal,
            smooth_sigma=smooth_sigma,
            window=local_window,
        )
    return encode_signal_wavelet(
        raw_signal,
        wavelet=wavelet,
        level=wavelet_level,
        window=local_window,
    )
