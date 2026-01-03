# Example Annotation Data

This directory contains small example BED files to demonstrate the format of annotation data used for training the replication fork and ORI detection models.

## Files

### `example_oris_curated.bed`
Example curated origin of replication (ORI) annotations (100 entries).

**Format:**
```
chr  start    end      read_id          score  strand
Chr1 12345    23456    read_001         .      .
```

### `example_left_forks.bed`
Example left replication fork annotations (50 entries).

**Format:**
```
chr  start    end      read_id          score  strand
Chr1 45678    56789    read_002         .      .
```

### `example_right_forks.bed`
Example right replication fork annotations (50 entries).

**Format:**
```
chr  start    end      read_id          score  strand
Chr2 78901    89012    read_003         .      .
```

## Usage

These files can be used as examples to understand the expected BED format for training:

```bash
# Train ORI model with your own data (using same format)
python scripts/train_ori_model.py \
    --config configs/ori_model_default.yaml

# Train Fork model with your own data
python scripts/train_fork_model.py \
    --config configs/fork_model_default.yaml
```

## Notes

- These are **example files** with a subset of real data (50-100 entries each)
- Full datasets typically contain thousands of annotations
- BED format is tab-separated with columns: chr, start, end, read_id, score, strand
- Files can be visualized in genome browsers (IGV, UCSC Genome Browser, etc.)
