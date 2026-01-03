"""
Prediction and inference utilities for trained models.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from ..data.encoding import encode_signal_enhanced


def predict_on_read(model, read_df, max_length, use_enhanced_encoding=True):
    """
    Predict on a single read.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model
    read_df : pd.DataFrame
        Read data with columns: start, end, signal, read_id
    max_length : int
        Maximum sequence length (from training)
    use_enhanced_encoding : bool
        Use 9-channel (True) or 6-channel (False) encoding

    Returns
    -------
    tuple
        (read_df, predictions) where predictions has shape (n_segments,) or (n_segments, n_classes)
    """
    from ..data.encoding import encode_signal_basic

    read_df = read_df.sort_values('start').reset_index(drop=True)

    if len(read_df) == 0:
        return read_df, None

    # Encode signal
    encoder = encode_signal_enhanced if use_enhanced_encoding else encode_signal_basic
    signal = read_df['signal'].values
    X_encoded = encoder(signal)

    # Pad
    X_padded = np.zeros((1, max_length, X_encoded.shape[1]), dtype=np.float32)
    use_length = min(len(X_encoded), max_length)
    X_padded[0, :use_length, :] = X_encoded[:use_length, :]

    # Predict
    y_pred_full = model.predict(X_padded, verbose=0)[0]

    # Trim to actual read length
    y_pred = y_pred_full[:len(read_df)]

    return read_df, y_pred


def predict_on_all_reads(model, xy_data, max_length, read_ids=None,
                         use_enhanced_encoding=True, verbose=True):
    """
    Predict on multiple reads.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model
    xy_data : pd.DataFrame
        XY signal data
    max_length : int
        Maximum sequence length
    read_ids : list, optional
        Specific read IDs to process (if None, process all)
    use_enhanced_encoding : bool
        Encoding type
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions for all segments
    """
    if read_ids is None:
        read_ids = xy_data['read_id'].unique()

    all_predictions = []

    for idx, read_id in enumerate(read_ids):
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(read_ids)} reads...")

        read_df = xy_data[xy_data['read_id'] == read_id].copy()
        read_df, y_pred = predict_on_read(model, read_df, max_length, use_enhanced_encoding)

        if y_pred is None:
            continue

        # Add predictions to dataframe
        if len(y_pred.shape) == 1:
            # Binary classification
            read_df['ori_prob'] = y_pred
            read_df['is_ori'] = (y_pred > 0.5).astype(int)
        else:
            # Multi-class classification
            read_df['class_0_prob'] = y_pred[:, 0]
            read_df['class_1_prob'] = y_pred[:, 1]
            read_df['class_2_prob'] = y_pred[:, 2]
            read_df['predicted_class'] = np.argmax(y_pred, axis=1)

        all_predictions.append(read_df)

    if len(all_predictions) == 0:
        print("⚠️ No predictions generated!")
        return None

    combined = pd.concat(all_predictions, ignore_index=True)

    if verbose:
        print(f"\n✅ Predictions complete: {len(combined):,} segments from {len(read_ids):,} reads")

    return combined


def call_peaks_from_predictions(predictions_df, threshold=0.5, min_length=100):
    """
    Call peaks (ORIs or forks) from segment-level predictions.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions
    threshold : float
        Probability threshold for calling peaks
    min_length : int
        Minimum peak length in bp

    Returns
    -------
    pd.DataFrame
        Called peaks with columns: chr, start, end, read_id, max_prob, length
    """
    if 'ori_prob' in predictions_df.columns:
        prob_col = 'ori_prob'
    else:
        # Multi-class: use max probability
        prob_cols = [c for c in predictions_df.columns if c.endswith('_prob')]
        predictions_df['max_prob'] = predictions_df[prob_cols].max(axis=1)
        prob_col = 'max_prob'

    peaks = []

    for read_id, read_df in predictions_df.groupby('read_id'):
        read_df = read_df.sort_values('start').reset_index(drop=True)

        # Find segments above threshold
        above_threshold = read_df[prob_col] > threshold

        if not above_threshold.any():
            continue

        # Find contiguous regions
        regions = []
        in_region = False
        region_start_idx = None

        for idx, is_above in enumerate(above_threshold):
            if is_above and not in_region:
                # Start of new region
                in_region = True
                region_start_idx = idx
            elif not is_above and in_region:
                # End of region
                in_region = False
                regions.append((region_start_idx, idx - 1))
            elif is_above and in_region and idx == len(above_threshold) - 1:
                # End of read while in region
                regions.append((region_start_idx, idx))

        # Convert to genomic coordinates
        for start_idx, end_idx in regions:
            region_df = read_df.iloc[start_idx:end_idx+1]

            peak_start = region_df['start'].min()
            peak_end = region_df['end'].max()
            peak_length = peak_end - peak_start

            if peak_length >= min_length:
                peaks.append({
                    'chr': region_df['chr'].iloc[0],
                    'start': peak_start,
                    'end': peak_end,
                    'read_id': read_id,
                    'max_prob': region_df[prob_col].max(),
                    'mean_prob': region_df[prob_col].mean(),
                    'length': peak_length,
                    'n_segments': len(region_df)
                })

    if len(peaks) == 0:
        print("⚠️ No peaks called with current threshold!")
        return pd.DataFrame()

    peaks_df = pd.DataFrame(peaks)
    print(f"✅ Called {len(peaks_df):,} peaks")

    return peaks_df


def export_peaks_to_bed(peaks_df, output_file, score_col='max_prob'):
    """
    Export called peaks to BED format.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Peaks dataframe
    output_file : str
        Output BED file path
    score_col : str
        Column to use for BED score field
    """
    bed_df = peaks_df[['chr', 'start', 'end', 'read_id']].copy()
    bed_df['score'] = (peaks_df[score_col] * 1000).astype(int)  # Scale to 0-1000
    bed_df['strand'] = '.'

    bed_df.to_csv(output_file, sep='\t', header=False, index=False)
    print(f"✅ BED file saved: {output_file}")
