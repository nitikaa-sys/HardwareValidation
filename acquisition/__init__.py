"""
Acquisition package for ADS1299 hardware validation system.

Modules:
- constants: Protocol constants and defaults
- config: Configuration loading and validation
- hardware: HardwareClient for ESP32/ADS1299 communication
- schedule: Event scheduling and randomization
- naming: Filename and sidecar metadata contracts
- streaming: Frame streaming and experiment execution

Usage:
    from acquisition import HardwareClient, run_config_experiment
    from acquisition.config import load_config, validate_experiment_config
    from acquisition.schedule import generate_event_schedule
"""

from pathlib import Path

# Package directory (for path-robust operations)
PACKAGE_DIR = Path(__file__).parent.resolve()

# Core exports
from .hardware import HardwareClient
from .streaming import (
    ExperimentEvent,
    run_config_experiment,
    run_experiment,
    stream_event,
    background_save,
    capture_live_stream
)
from .config import (
    load_config,
    validate_experiment_config,
    merge_config_with_args,
    get_default_config_path,
    get_profiles_dir
)
from .schedule import (
    setup_random_seed,
    generate_event_schedule,
    display_event_schedule
)
from .naming import (
    compute_output_folder,
    build_condition_code,
    build_sidecar_metadata,
    build_filename
)

# Version
__version__ = "2.0.0"

__all__ = [
    # Hardware
    "HardwareClient",
    
    # Streaming
    "ExperimentEvent",
    "run_config_experiment",
    "run_experiment",
    "stream_event",
    "background_save",
    "capture_live_stream",
    
    # Config
    "load_config",
    "validate_experiment_config",
    "merge_config_with_args",
    "get_default_config_path",
    "get_profiles_dir",
    
    # Schedule
    "setup_random_seed",
    "generate_event_schedule",
    "display_event_schedule",
    
    # Naming
    "compute_output_folder",
    "build_condition_code",
    "build_sidecar_metadata",
    "build_filename",
    
    # Paths
    "PACKAGE_DIR",
]
