"""
Replication Analyzer - Deep Learning for ORI and Fork Detection

A modular package for detecting replication origins (ORIs) and replication
forks in BrdU/EdU labeled DNA sequencing data.
"""

__version__ = "0.1.0"
__author__ = "Crisanto Project"

# Expose key functions at package level
from .data.loaders import (
    load_curated_origins,
    load_all_xy_data,
    load_fork_data,
    load_genomic_regions
)

from .data.encoding import (
    encode_signal_enhanced,
    encode_signal_basic
)

from .data.preprocessing import (
    prepare_ori_data_hybrid,
    prepare_fork_data_hybrid,
    pad_sequences
)

# Model imports are intentionally NOT here — importing them triggers TensorFlow
# initialization which consumes ~400-800 MB even in preprocessing-only runs.
# Import models directly from their submodules when needed:
#   from replication_analyzer.models.fork_model import build_fork_detection_model
#   from replication_analyzer.models.ori_model import build_ori_expert_model
#   from replication_analyzer.models.losses import FocalLoss

__all__ = [
    # Data loading
    'load_curated_origins',
    'load_all_xy_data',
    'load_fork_data',
    'load_genomic_regions',
    # Encoding
    'encode_signal_enhanced',
    'encode_signal_basic',
    # Preprocessing
    'prepare_ori_data_hybrid',
    'prepare_fork_data_hybrid',
    'pad_sequences',
]
