"""
Analysis Package

Offline analysis pipeline for EEG binary data files.
Extracts raw ADC counts, performs timing integrity checks,
and generates comprehensive PDF reports.

Usage:
    # Hardware validation (single file)
    from analysis.pipeline import run_pipeline
    results = run_pipeline(Path("data.bin"), export_pdf=True)

    # Eyes Open vs Eyes Closed comparison
    from analysis.decode_bin import parse_framestream_bin
    counts_eo, meta_eo = parse_framestream_bin(bin_bytes)
"""

__version__ = "2.3.0"
__author__ = "Hardware Validation Team"

# Primary API - Hardware Validation
from .pipeline import run_pipeline

# Core functions for direct access
from .parse_filename import parse_filename_metadata, detect_test_type
from .decode_bin import (
    parse_ads1299_framestream_bin_bytes_strict_1416,  # Legacy name (keep for compatibility)
    parse_framestream_bin,                             # Clean alias (use this)
    duration_from_counts,
)
from .preprocess import counts_to_uv, preprocess_channel_uv, FS_HZ, VREF_V, GAIN

# EEG electrode mapping and plots
from .plots import (
    ELECTRODE_MAP,
    FUNCTIONAL_CHANNELS,
    plot_eo_ec_montage,
    plot_eo_ec_psd_comparison,
    plot_eo_ec_publication,
    plot_eo_ec_publication_montage,
    plot_eo_ec_publication_psd,
    plot_eo_ec_publication_complete,
)

__all__ = [
    # Pipeline
    "run_pipeline",
    # Filename parsing
    "parse_filename_metadata",
    "detect_test_type",
    # Binary decoding
    "parse_framestream_bin",                             # Use this in new code
    "parse_ads1299_framestream_bin_bytes_strict_1416",   # Legacy compatibility
    "duration_from_counts",
    # Signal processing
    "counts_to_uv",
    "preprocess_channel_uv",
    # Constants
    "FS_HZ",
    "VREF_V",
    "GAIN",
    # EEG electrode mapping
    "ELECTRODE_MAP",
    "FUNCTIONAL_CHANNELS",
    # Plots
    "plot_eo_ec_montage",
    "plot_eo_ec_psd_comparison",
    "plot_eo_ec_publication",
    "plot_eo_ec_publication_montage",
    "plot_eo_ec_publication_psd",
    "plot_eo_ec_publication_complete",
]
