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

from .models.ori_model import build_ori_expert_model, build_ori_simple_model
from .models.fork_model import build_fork_detection_model
from .models.losses import FocalLoss, MultiClassFocalLoss

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
    # Models
    'build_ori_expert_model',
    'build_ori_simple_model',
    'build_fork_detection_model',
    # Losses
    'FocalLoss',
    'MultiClassFocalLoss',
]
