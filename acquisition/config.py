"""
Configuration loading and validation for ADS1299 acquisition system.

Functions:
- load_config() - Read JSON config file
- validate_experiment_config() - Validate all sections
- validate_condition_mapping() - Sanitize condition labels
- merge_config_with_args() - CLI argument override
- calculate_frames_from_duration() - Duration -> frames math
- extract_sample_rate_from_profile() - Read sample rate from profile JSON
"""

import json
import os
import argparse
from pathlib import Path
from typing import Optional

from .constants import SAMPLES_PER_FRAME, DEFAULT_SAMPLE_RATE

# Resolve module directory for path-robust operations
MODULE_DIR = Path(__file__).parent.resolve()


def calculate_frames_from_duration(sample_rate: int, duration_seconds: float) -> int:
    """
    Calculate number of frames needed for a given duration and sample rate.
    
    Formula: frames = (sample_rate × duration_seconds) / SAMPLES_PER_FRAME
    
    Args:
        sample_rate: Sampling rate in Hz (e.g., 16000, 1000)
        duration_seconds: Desired acquisition duration in seconds
        
    Returns:
        Number of frames needed (rounded to nearest integer)
    """
    total_samples = sample_rate * duration_seconds
    frames = int(round(total_samples / SAMPLES_PER_FRAME))
    return frames


def extract_sample_rate_from_profile(profile_path: str) -> int:
    """
    Extract sample rate from a profile JSON file.
    
    Args:
        profile_path: Path to profile JSON file
        
    Returns:
        Sample rate in Hz (default 16000 if not found)
    """
    try:
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        sample_rate = profile.get('CONFIG1_SAMPLE_RATE_SPS', DEFAULT_SAMPLE_RATE)
        return sample_rate
    except Exception as e:
        print(f"    ⚠ Could not read sample rate from profile: {e}")
        print(f"    Using default: {DEFAULT_SAMPLE_RATE} SPS")
        return DEFAULT_SAMPLE_RATE


def load_config(config_path: str) -> dict:
    """
    Load experiment configuration from JSON file.
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If config file is invalid or missing required fields
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[CONFIG] Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read config file: {e}")


def validate_condition_mapping(mapping: dict, allow_duplicates: bool = False) -> dict:
    """
    Validate condition mapping and sanitize labels for filenames.
    
    Args:
        mapping: Dict mapping LED colors to condition labels
        allow_duplicates: If True, skip duplicate label check (for validation mode)
        
    Returns:
        Sanitized mapping dict
        
    Raises:
        ValueError: If mapping is invalid
    """
    REQUIRED_COLORS = {"RED", "BLUE", "GREEN", "YELLOW"}
    
    if not isinstance(mapping, dict):
        raise ValueError("conditions.mapping must be a dictionary")
    
    # Check all required colors present
    provided_colors = set(mapping.keys())
    if provided_colors != REQUIRED_COLORS:
        missing = REQUIRED_COLORS - provided_colors
        extra = provided_colors - REQUIRED_COLORS
        error_parts = []
        if missing:
            error_parts.append(f"Missing colors: {sorted(missing)}")
        if extra:
            error_parts.append(f"Extra colors: {sorted(extra)}")
        raise ValueError(
            f"conditions.mapping must define exactly RED, BLUE, GREEN, YELLOW.\n"
            f"   {', '.join(error_parts)}"
        )
    
    # Check for duplicate labels (skip in validation mode)
    if not allow_duplicates:
        labels = list(mapping.values())
        if len(labels) != len(set(labels)):
            duplicates = [l for l in labels if labels.count(l) > 1]
            raise ValueError(
                f"Duplicate condition labels found: {set(duplicates)}\n"
                f"   Each LED color must map to a unique condition label."
            )
    
    # Sanitize labels for filenames
    sanitized = {}
    for color, label in mapping.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Condition label for {color} must be a non-empty string")
        
        # Replace spaces with underscores, remove/replace special chars
        clean_label = label.strip().replace(' ', '_')
        clean_label = ''.join(c for c in clean_label if c.isalnum() or c in '_-')
        
        if not clean_label:
            raise ValueError(f"Condition label for {color} contains no valid characters: '{label}'")
        
        sanitized[color] = clean_label
        
        if clean_label != label.strip():
            print(f"    [SANITIZE] '{label}' → '{clean_label}' (filesystem safe)")
    
    return sanitized


def validate_experiment_config(config: dict) -> None:
    """
    Validate experiment configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check top-level structure
    if 'esp32' not in config:
        raise ValueError("Missing required section: esp32")
    if 'experiment' not in config:
        raise ValueError("Missing required section: experiment")
    if 'routine' not in config:
        raise ValueError("Missing required section: routine")
    if 'conditions' not in config:
        raise ValueError("Missing required section: conditions")
    
    # Validate esp32 section
    if 'ip' not in config['esp32']:
        raise ValueError("Missing required field: esp32.ip")
    if not isinstance(config['esp32']['ip'], str) or not config['esp32']['ip'].strip():
        raise ValueError("esp32.ip must be a non-empty string")
    
    # Validate experiment section
    if 'num_events' not in config['experiment']:
        raise ValueError("Missing required field: experiment.num_events")
    
    num_events = config['experiment']['num_events']
    if not isinstance(num_events, int) or num_events <= 0:
        raise ValueError(f"experiment.num_events must be a positive integer, got: {num_events}")
    
    # Validate random_seed (optional)
    if 'random_seed' in config['experiment']:
        seed = config['experiment']['random_seed']
        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"experiment.random_seed must be an integer or null, got: {type(seed).__name__}")
    
    # Validate profile_path (optional)
    if 'profile_path' in config['experiment'] and config['experiment']['profile_path'] is not None:
        profile_path = config['experiment']['profile_path']
        if not isinstance(profile_path, str):
            raise ValueError(f"experiment.profile_path must be a string or null")
        if profile_path and not os.path.exists(profile_path):
            raise ValueError(f"Profile file not found: {profile_path}")
    
    # Validate output_folder (optional, default to "experiments")
    if 'output_folder' in config['experiment']:
        if not isinstance(config['experiment']['output_folder'], str):
            raise ValueError("experiment.output_folder must be a string")
    
    # Validate inter_event_delay (optional, default to 0.5)
    if 'inter_event_delay' in config['experiment']:
        delay = config['experiment']['inter_event_delay']
        if not isinstance(delay, (int, float)) or delay < 0:
            raise ValueError(f"experiment.inter_event_delay must be >= 0, got: {delay}")
    
    # Validate routine section - support both acquisition_duration_seconds (new) and frames_per_event (legacy)
    acquisition_duration = config['routine'].get('acquisition_duration_seconds')
    frames_per_event = config['routine'].get('frames_per_event')
    
    # Check that at least one is provided
    if acquisition_duration is None and frames_per_event is None:
        raise ValueError(
            "Missing required field in routine section.\n"
            "   Provide either:\n"
            "   - acquisition_duration_seconds (recommended, duration-based)\n"
            "   - frames_per_event (legacy, frame-based)"
        )
    
    # If acquisition_duration_seconds is provided, calculate frames automatically
    if acquisition_duration is not None:
        # Validate duration
        if not isinstance(acquisition_duration, (int, float)) or acquisition_duration <= 0:
            raise ValueError(f"routine.acquisition_duration_seconds must be a positive number, got: {acquisition_duration}")
        
        # Extract sample rate from profile (if specified)
        profile_path = config['experiment'].get('profile_path')
        if profile_path:
            sample_rate = extract_sample_rate_from_profile(profile_path)
            print(f"    [AUTO-CALC] Profile sample rate: {sample_rate} SPS")
        else:
            sample_rate = DEFAULT_SAMPLE_RATE
            print(f"    [AUTO-CALC] No profile specified, using default: {sample_rate} SPS")
        
        # Calculate frames
        calculated_frames = calculate_frames_from_duration(sample_rate, acquisition_duration)
        
        print(f"    [AUTO-CALC] Duration: {acquisition_duration}s × {sample_rate} SPS ÷ 50 = {calculated_frames} frames")
        
        # Set calculated value in config (overrides frames_per_event if both provided)
        config['routine']['frames_per_event'] = calculated_frames
        
        if frames_per_event is not None and frames_per_event != calculated_frames:
            print(f"    [AUTO-CALC] ⚠ Overriding frames_per_event ({frames_per_event}) with calculated value ({calculated_frames})")
    
    # Validate final frames_per_event value
    frames_per_event = config['routine']['frames_per_event']
    if not isinstance(frames_per_event, int) or frames_per_event <= 0:
        raise ValueError("routine.frames_per_event must be a positive integer")
    
    if 'sound_duration_us' not in config['routine']:
        raise ValueError("Missing required field: routine.sound_duration_us")
    if not isinstance(config['routine']['sound_duration_us'], int) or config['routine']['sound_duration_us'] < 0:
        raise ValueError("routine.sound_duration_us must be >= 0")
    
    if 'led_enabled' not in config['routine']:
        raise ValueError("Missing required field: routine.led_enabled")
    if not isinstance(config['routine']['led_enabled'], bool):
        raise ValueError("routine.led_enabled must be true or false")
    
    # Validate conditions section
    if 'enforce_equal_condition_count' not in config['conditions']:
        raise ValueError("Missing required field: conditions.enforce_equal_condition_count")
    if not isinstance(config['conditions']['enforce_equal_condition_count'], bool):
        raise ValueError("conditions.enforce_equal_condition_count must be true or false")
    
    if 'mapping' not in config['conditions']:
        raise ValueError("Missing required field: conditions.mapping")
    
    # Check if validation mode (allows duplicate labels like "VAL" for all colors)
    is_validation_mode = config.get('meta', {}).get('mode') == 'validation'
    
    # Validate and sanitize mapping (will raise ValueError if invalid)
    config['conditions']['mapping'] = validate_condition_mapping(
        config['conditions']['mapping'],
        allow_duplicates=is_validation_mode
    )
    
    # Validate equal condition count divisibility
    if config['conditions']['enforce_equal_condition_count']:
        num_conditions = len(config['conditions']['mapping'])
        if num_events % num_conditions != 0:
            raise ValueError(
                f"With enforce_equal_condition_count=true and {num_conditions} conditions, "
                f"num_events must be divisible by {num_conditions}. Got {num_events} events."
            )
    
    print(f"    ✓ Config validated successfully")


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """
    Merge configuration from file with command-line arguments.
    CLI arguments take precedence over config file.
    
    Args:
        config: Configuration dictionary from file
        args: Parsed command-line arguments
        
    Returns:
        Merged configuration dictionary
    """
    # CLI args override config values
    if hasattr(args, 'events') and args.events is not None:
        config['experiment']['num_events'] = args.events
        print(f"    [OVERRIDE] num_events = {args.events} (from CLI)")
    
    if hasattr(args, 'ip') and args.ip != 'node.local':  # Only override if explicitly set
        config['esp32']['ip'] = args.ip
        print(f"    [OVERRIDE] esp32.ip = {args.ip} (from CLI)")
    
    if hasattr(args, 'profile') and args.profile is not None:
        config['experiment']['profile_path'] = args.profile
        print(f"    [OVERRIDE] profile_path = {args.profile} (from CLI)")
    
    return config


def get_default_config_path() -> Path:
    """
    Get the default config.json path (relative to module directory).
    
    Returns:
        Path to config.json in module directory
    """
    return MODULE_DIR / "config.json"


def get_profiles_dir() -> Path:
    """
    Get the profiles directory path (relative to module directory).
    
    Returns:
        Path to profiles/ directory
    """
    return MODULE_DIR / "profiles"
