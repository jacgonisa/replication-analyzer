"""Read-level split manifests with no oversampling leakage."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def build_read_metadata(
    xy_data: pd.DataFrame,
    left_forks: pd.DataFrame,
    right_forks: pd.DataFrame,
    origins: pd.DataFrame,
    terminations: pd.DataFrame,
) -> pd.DataFrame:
    """Collect one metadata row per read for split stratification."""
    read_lengths = xy_data.groupby("read_id").size().rename("n_windows")
    read_chr = xy_data.groupby("read_id")["chr"].first().rename("chr")

    metadata = pd.DataFrame({"read_id": xy_data["read_id"].unique()})
    metadata = metadata.merge(read_lengths, on="read_id", how="left")
    metadata = metadata.merge(read_chr, on="read_id", how="left")

    for name, df in [
        ("has_left", left_forks),
        ("has_right", right_forks),
        ("has_origin", origins),
        ("has_termination", terminations),
    ]:
        present = pd.Series(True, index=pd.Index(df["read_id"].unique(), name="read_id"), name=name)
        metadata = metadata.merge(present, on="read_id", how="left")
        metadata[name] = metadata[name].fillna(False)

    metadata["has_any_event"] = metadata[
        ["has_left", "has_right", "has_origin", "has_termination"]
    ].any(axis=1)
    metadata["stratify_label"] = metadata.apply(
        lambda row: "".join(
            [
                "1" if row["has_left"] else "0",
                "1" if row["has_right"] else "0",
                "1" if row["has_origin"] else "0",
                "1" if row["has_termination"] else "0",
            ]
        ),
        axis=1,
    )

    rare_labels = metadata["stratify_label"].value_counts()
    metadata.loc[
        metadata["stratify_label"].map(rare_labels) < 2,
        "stratify_label",
    ] = metadata["has_any_event"].map({True: "event", False: "background"})
    return metadata


def create_split_manifest(
    metadata: pd.DataFrame,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create train/val/test partitions by read_id."""
    if test_fraction < 0 or val_fraction <= 0 or (val_fraction + test_fraction) >= 1:
        raise ValueError("Invalid split fractions.")

    read_ids = metadata["read_id"].tolist()
    holdout_fraction = val_fraction + test_fraction
    holdout_size = max(1, int(round(len(metadata) * holdout_fraction)))

    def _choose_stratify_labels(frame: pd.DataFrame) -> Optional[pd.Series]:
        candidates = [
            frame["stratify_label"],
            frame["has_any_event"].map({True: "event", False: "background"}),
            None,
        ]
        for labels in candidates:
            if labels is None:
                return None
            counts = labels.value_counts()
            if counts.empty:
                continue
            # train_test_split stratification requires at least 2 per class and
            # enough holdout slots to place at least one example from each class.
            if counts.min() < 2:
                continue
            if len(counts) > holdout_size:
                continue
            return labels
        return None

    stratify_labels = _choose_stratify_labels(metadata)

    train_ids, holdout_ids = train_test_split(
        read_ids,
        test_size=holdout_fraction,
        random_state=random_seed,
        stratify=stratify_labels,
    )

    manifest = metadata.copy()
    manifest["split"] = "train"

    if test_fraction > 0:
        holdout_metadata = metadata[metadata["read_id"].isin(holdout_ids)].copy()
        holdout_test_fraction = test_fraction / holdout_fraction
        holdout_stratify = _choose_stratify_labels(holdout_metadata)
        val_ids, test_ids = train_test_split(
            holdout_ids,
            test_size=holdout_test_fraction,
            random_state=random_seed,
            stratify=holdout_stratify,
        )
        manifest.loc[manifest["read_id"].isin(val_ids), "split"] = "val"
        manifest.loc[manifest["read_id"].isin(test_ids), "split"] = "test"
    else:
        manifest.loc[manifest["read_id"].isin(holdout_ids), "split"] = "val"

    return manifest


def save_split_manifest(manifest: pd.DataFrame, output_path: str) -> None:
    """Save split manifest to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, sep="\t", index=False)


def load_split_manifest(path: str) -> pd.DataFrame:
    """Load split manifest from disk."""
    return pd.read_csv(path, sep="\t")
