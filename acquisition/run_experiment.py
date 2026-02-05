#!/usr/bin/env python3
"""
ADS1299 Experiment Runner Script
Runs multiple EEG acquisition events with random LED colors and sound

This script performs a series of EEG data acquisition events:
- Each event streams 1665 frames (83250 samples) with sound for 5 seconds
- Random LED color is displayed for each event
- Binary files are saved to the output folder
- Failures are logged and the experiment continues

Configuration:
  FRAMES_PER_EVENT:       Number of frames per event (default: 1665)
  SOUND_DURATION_US:      Sound duration in µs (default: 5000000 = 5s)
  OUTPUT_FOLDER:          Folder for binary output files
  DELAY_BETWEEN_EVENTS:   Delay between events in seconds

Usage:
  python run_experiment.py --events 10                    # Run 10 events
  python run_experiment.py --events 50 --ip 192.168.1.100 # Custom IP
  python run_experiment.py --events 20 --profile abr.json # Load profile first
"""

import asyncio
import websockets
import requests
import struct
import time
import random
import os
from datetime import datetime
import sys
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

# ============ EXPERIMENT CONFIGURATION ============
FRAMES_PER_EVENT = 1665          # Fixed: 1665 frames per event (83250 samples)
SOUND_DURATION_US = 5000000      # Fixed: 5 seconds (5,000,000 µs)
OUTPUT_FOLDER = "experiments"    # Folder for binary output files
INTER_EVENT_DELAY = 0.5          # Delay between events for hardware reset (seconds)
VALID_COLORS = ['RED', 'BLUE', 'GREEN', 'YELLOW']  # Colors for random selection
MAX_SAVE_WORKERS = 4             # Max parallel background save threads
# ==================================================

# ============ CONDITION CODE MAPPINGS (Parser-Compatible) ============
VALIDATION_SCENARIO_CODES = {
    "general_test": "GEN",
    "internal_noise": "INT",
    "external_noise": "EXT",
    "known_signal_injection": "KSI",
    "functional_tests": "FUN",
    "microphone_tests": "MIC"
}

EXPERIMENT_CONDITION_CODES = {
    "LEFT": "LEF",
    "RIGHT": "RIG",
    "MATH": "MTH",
    "REST": "RST"
}
# =====================================================================

# Frame format constants (matching C backend)
FRAME_HEADER_SIZE = 16      # 16-byte header
PACKET_SIZE = 28            # 27 data + 1 DMA padding
PACKET_DATA_SIZE = 27       # 3 status + 24 channel data
FRAME_TARGET_SIZE = 1416    # 16 + (50 × 28)
PACKETS_PER_FRAME = 50      # Fixed: 50 packets per frame
SAMPLES_PER_FRAME = 50      # Each packet = 1 sample
CHANNELS_PER_SAMPLE = 8     # ADS1299 has 8 channels

TIMEOUT_NO_DATA = 10.0      # Stop if no data for 10 seconds

# Global state
ESP32_IP = "node.local"
HTTP_URL = ""
WS_URL = ""
SAMPLE_RATE = 0  # Extracted from register dump


# ============ CONFIG LOADING & VALIDATION ============

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
        sample_rate = profile.get('CONFIG1_SAMPLE_RATE_SPS', 16000)
        return sample_rate
    except Exception as e:
        print(f"    ⚠ Could not read sample rate from profile: {e}")
        print(f"    Using default: 16000 SPS")
        return 16000


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
            sample_rate = 16000  # Default
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


# ============ REPRODUCIBILITY & SCHEDULING ============

def setup_random_seed(config: dict) -> None:
    """
    Set random seed for reproducible experiments.
    
    Args:
        config: Configuration dictionary
    """
    seed = config.get('experiment', {}).get('random_seed', None)
    
    if seed is not None:
        random.seed(seed)
        print(f"\n[REPRODUCIBILITY] Random seed set to: {seed}")
        print(f"                  Experiment will produce identical event order on re-runs")
    else:
        print(f"\n[REPRODUCIBILITY] No seed specified - using non-deterministic randomization")


def generate_event_schedule(num_events: int, condition_mapping: dict, enforce_equal_count: bool) -> list:
    """
    Generate randomized event schedule with optional balance enforcement.
    
    Args:
        num_events: Number of events to generate
        condition_mapping: Dict mapping LED colors to condition labels
        enforce_equal_count: If True, ensure equal counts of each condition
        
    Returns:
        List of (led_color, condition_label) tuples
        
    Raises:
        ValueError: If enforce_equal_count=True and num_events not divisible by 4
    """
    BASE_COLORS = ["RED", "BLUE", "GREEN", "YELLOW"]
    
    if enforce_equal_count:
        # Enforce equal count: equal counts, then shuffle
        if num_events % 4 != 0:
            raise ValueError(
                f"With enforce_equal_condition_count=true and 4 conditions, "
                f"num_events must be divisible by 4. Got {num_events}."
            )
        
        reps_per_color = num_events // 4
        color_list = []
        for color in BASE_COLORS:
            color_list.extend([color] * reps_per_color)
        
        # Shuffle once
        random.shuffle(color_list)
    
    else:
        # Non-enforced: random choice per event (already random, no shuffle)
        color_list = [random.choice(BASE_COLORS) for _ in range(num_events)]
    
    # Map colors to condition labels
    schedule = [(color, condition_mapping[color]) for color in color_list]
    
    return schedule


def display_event_schedule(schedule: list, enforce_equal_count: bool, seed: int = None) -> None:
    """
    Display the complete event schedule to the user for review.
    
    Args:
        schedule: List of (led_color, condition_label) tuples
        enforce_equal_count: Whether balance enforcement was used
        seed: Random seed used (None if not specified)
    """
    num_events = len(schedule)
    
    print(f"\n[EVENT SCHEDULE] {num_events} events generated")
    print(f"                 Equal condition count: {enforce_equal_count}")
    if seed is not None:
        print(f"                 Random seed: {seed}")
    
    # Show first 10 and last 5 events (or all if <= 20)
    if num_events <= 20:
        # Show all events
        for i, (color, label) in enumerate(schedule, 1):
            print(f"    Event {i:3d}: {color:6s} → {label}")
    else:
        # Show first 10
        for i, (color, label) in enumerate(schedule[:10], 1):
            print(f"    Event {i:3d}: {color:6s} → {label}")
        print(f"    ...")
        # Show last 5
        for i, (color, label) in enumerate(schedule[-5:], num_events - 4):
            print(f"    Event {i:3d}: {color:6s} → {label}")
    
    # Calculate and display balance statistics
    print(f"\n    Balance verification:")
    label_counts = {}
    for color, label in schedule:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / num_events) * 100
        print(f"        {label:15s}: {count:3d} events ({percentage:5.1f}%)")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='ADS1299 Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Default: loads config.json automatically
  %(prog)s
  
  # Use custom config file
  %(prog)s --config my_experiment.json
  
  # Override specific config values
  %(prog)s --events 20                     # Override num_events
  %(prog)s --events 50 --ip 192.168.1.100  # Override events + ESP32 IP
  %(prog)s --profile custom.json           # Override ADS1299 profile

Configuration:
  - By default, loads ./config.json
  - CLI flags (--events, --ip, --profile) override config values
  - Supports balanced experimental design with reproducible random seeds
  - Falls back to simple CLI mode if config.json missing and --events provided
        '''
    )
    
    parser.add_argument('-c', '--config', type=str, default=None,
                       help='Path to config file (default: config.json)')
    parser.add_argument('-n', '--events', type=int, default=None,
                       help='Number of events (overrides config value)')
    parser.add_argument('-i', '--ip', '--host', dest='ip', type=str, default='node.local',
                       help='ESP32 IP or hostname (overrides config, default: node.local)')
    parser.add_argument('-p', '--profile', type=str, default=None,
                       help='ADS1299 profile JSON file (overrides config value)')
    
    return parser.parse_args()


def dump_and_verify_registers() -> dict:
    """
    Dump all ADS1299 registers and display for user verification.
    
    Calls GET /ads1299/diagnostics/registers-dump to read all 24 ADS1299 registers.
    
    Returns:
        Dictionary of register values if successful, empty dict on failure
        
    Side Effects:
        Sets global SAMPLE_RATE based on CONFIG1 register
    """
    global SAMPLE_RATE
    print(f"\n    Reading ADS1299 hardware registers...")
    
    try:
        response = requests.get(
            f"{HTTP_URL}/ads1299/diagnostics/registers-dump",
            timeout=10
        )
        if response.status_code == 200:
            registers = response.json()
            
            # Sample rate lookup from CONFIG1 ODR bits
            sample_rate_map = {
                0: 16000, 1: 8000, 2: 4000, 3: 2000,
                4: 1000, 5: 500, 6: 250
            }
            
            # Parse CONFIG1 for sample rate
            config1_hex = registers.get('CONFIG1', '0x00')
            config1_val = int(config1_hex, 16) if isinstance(config1_hex, str) else config1_hex
            odr_bits = config1_val & 0x07
            SAMPLE_RATE = sample_rate_map.get(odr_bits, 0)
            sample_rate = str(SAMPLE_RATE) if SAMPLE_RATE > 0 else "unknown"
            
            # Parse CONFIG3 for internal reference
            config3_hex = registers.get('CONFIG3', '0x00')
            config3_val = int(config3_hex, 16) if isinstance(config3_hex, str) else config3_hex
            int_ref = "enabled" if (config3_val & 0x80) else "disabled"
            
            # Display register dump
            print(f"    ┌─────────────────────────────────────────────────┐")
            print(f"    │  ADS1299 Hardware Register Dump                 │")
            print(f"    ├─────────────────────────────────────────────────┤")
            print(f"    │  ID       = {registers.get('ID', 'N/A'):>6}  (Device ID)              │")
            print(f"    │  CONFIG1  = {registers.get('CONFIG1', 'N/A'):>6}  ({sample_rate} SPS)             │")
            print(f"    │  CONFIG2  = {registers.get('CONFIG2', 'N/A'):>6}  (Test signal config)     │")
            print(f"    │  CONFIG3  = {registers.get('CONFIG3', 'N/A'):>6}  (Int ref {int_ref})     │")
            print(f"    │  LOFF     = {registers.get('LOFF', 'N/A'):>6}  (Lead-off detect)        │")
            print(f"    ├─────────────────────────────────────────────────┤")
            
            # Channel settings
            for i in range(1, 9):
                ch_key = f'CH{i}SET'
                ch_hex = registers.get(ch_key, '0x00')
                ch_val = int(ch_hex, 16) if isinstance(ch_hex, str) else ch_hex
                gain_bits = (ch_val >> 4) & 0x07
                gain_map = {0: "1", 1: "2", 2: "4", 3: "6", 4: "8", 5: "12", 6: "24"}
                gain = gain_map.get(gain_bits, "?")
                pd = "OFF" if (ch_val & 0x80) else "ON"
                print(f"    │  {ch_key}  = {ch_hex:>6}  (Gain={gain:>2}x, {pd:>3})         │")
            
            print(f"    ├─────────────────────────────────────────────────┤")
            print(f"    │  BIAS_SENSP = {registers.get('BIAS_SENSP', 'N/A'):>6}                        │")
            print(f"    │  BIAS_SENSN = {registers.get('BIAS_SENSN', 'N/A'):>6}                        │")
            print(f"    │  LOFF_SENSP = {registers.get('LOFF_SENSP', 'N/A'):>6}                        │")
            print(f"    │  LOFF_SENSN = {registers.get('LOFF_SENSN', 'N/A'):>6}                        │")
            print(f"    └─────────────────────────────────────────────────┘")
            
            # Verify device ID
            device_id = registers.get('ID', '0x00')
            if device_id in ['0x3E', '0x3e']:
                print(f"    ✓ Device verified: ADS1299 (ID={device_id})")
            else:
                print(f"    ⚠ Unexpected device ID: {device_id} (expected 0x3E for ADS1299)")
            
            return registers
        else:
            print(f"    ✗ Register dump failed: HTTP {response.status_code}")
            print(f"    Response: {response.text}")
            return {}
    except requests.exceptions.RequestException as e:
        print(f"    ✗ HTTP request failed: {e}")
        return {}


def convert_file_to_api_format(file_profile: dict) -> dict:
    """
    Convert file format profile to API format expected by PUT /ads1299/profile.
    
    File format uses field names like:
        CONFIG1_SAMPLE_RATE_SPS, CONFIG2_INT_REF_ENABLE, CH1_MODE, CH1_MUXP, etc.
    
    API format expects:
        sample_rate_sps, int_ref_enable, channels[] array with objects
    """
    api_profile = {
        "sample_rate_sps": file_profile.get("CONFIG1_SAMPLE_RATE_SPS", 16000),
        "int_ref_enable": file_profile.get("CONFIG2_INT_REF_ENABLE", True),
        "bias_meas_enable": file_profile.get("CONFIG3_BIAS_MEAS_ENABLE", False),
        "loff_magnitude_ua": file_profile.get("LOFF_MAGNITUDE_uA", 6),
        "loff_freq_hz": file_profile.get("LOFF_FREQ_HZ", 32),
        "chop_enable": file_profile.get("CHOP_ENABLE", False),
        "channels": []
    }
    
    # Convert CH1-CH8 to channels array
    for i in range(1, 9):
        channel = {
            "mode": file_profile.get(f"CH{i}_MODE", "EEG"),
            "muxp": file_profile.get(f"CH{i}_MUXP", f"IN{i}P"),
            "muxn": file_profile.get(f"CH{i}_MUXN", "SRB1"),
            "gain": file_profile.get(f"CH{i}_GAIN", 24),
            "enable": file_profile.get(f"CH{i}_ENABLE", True)
        }
        api_profile["channels"].append(channel)
    
    return api_profile


def load_and_apply_profile(profile_path: str) -> bool:
    """
    Load an ADS1299 profile from a local JSON file and apply it to the ESP32.
    
    Args:
        profile_path: Path to the local JSON profile file
        
    Returns:
        True if profile was loaded and applied successfully, False otherwise
    """
    print(f"\n[PROFILE] Loading ADS1299 profile from: {profile_path}")
    
    # Step 1: Read local JSON file
    try:
        with open(profile_path, 'r') as f:
            file_profile = json.load(f)
        print(f"    ✓ Profile file loaded ({len(file_profile)} settings)")
        
        # Display key settings
        sample_rate = file_profile.get('CONFIG1_SAMPLE_RATE_SPS', 'N/A')
        int_ref = file_profile.get('CONFIG2_INT_REF_ENABLE', 'N/A')
        print(f"    Sample rate: {sample_rate} SPS")
        print(f"    Internal ref: {int_ref}")
        
        # Count enabled channels
        enabled_channels = sum(1 for i in range(1, 9) if file_profile.get(f'CH{i}_ENABLE', False))
        print(f"    Enabled channels: {enabled_channels}/8")
        
    except FileNotFoundError:
        print(f"    ✗ Profile file not found: {profile_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"    ✗ Invalid JSON in profile file: {e}")
        return False
    except Exception as e:
        print(f"    ✗ Failed to read profile file: {e}")
        return False
    
    # Step 2: Convert file format to API format
    print(f"    Converting profile to API format...")
    api_profile = convert_file_to_api_format(file_profile)
    print(f"    ✓ Converted ({len(api_profile['channels'])} channels configured)")
    
    # Step 3: Send profile to ESP32 via PUT /ads1299/profile
    try:
        print(f"    Uploading profile to ESP32...")
        response = requests.put(
            f"{HTTP_URL}/ads1299/profile",
            json=api_profile,
            timeout=10
        )
        if response.status_code in [200, 204]:
            print(f"    ✓ Profile uploaded successfully")
        else:
            print(f"    ✗ Profile upload failed: HTTP {response.status_code}")
            print(f"    Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"    ✗ HTTP request failed: {e}")
        return False
    
    # Step 4: Reload profile to apply to hardware registers
    try:
        print(f"    Applying profile to ADS1299 hardware...")
        response = requests.post(
            f"{HTTP_URL}/ads1299/profile/reload",
            timeout=10
        )
        if response.status_code in [200, 202]:
            print(f"    ✓ Profile applied to ADS1299 hardware registers")
            return True
        else:
            print(f"    ✗ Profile reload failed: HTTP {response.status_code}")
            print(f"    Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"    ✗ HTTP request failed: {e}")
        return False


# ============ FOLDER & FILENAME HELPERS (Parser-Compatible) ============

def compute_output_folder(config: dict) -> str:
    """
    Compute output folder based on meta section.
    
    Folder structure:
    - validation: data/<session_name>/validation/
    - experiment: data/<session_name>/experiments/<subject_id>/
    - legacy: data/<session_name>/
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Full path to output folder
    """
    session_name = config['experiment'].get('output_folder', 'default_session')
    meta = config.get('meta')
    
    if not meta or not meta.get('mode'):
        # Legacy mode: data/<session_name>/
        return os.path.join('data', session_name)
    
    mode = meta['mode']
    
    if mode == 'validation':
        # Validation: data/<session_name>/validation/
        return os.path.join('data', session_name, 'validation')
    
    elif mode == 'experiment':
        # Experiment: data/<session_name>/experiments/<subject_id>/
        subject_id = meta.get('subject_id', 'unknown_subject')
        return os.path.join('data', session_name, 'experiments', subject_id)
    
    else:
        # Unknown mode → legacy
        return os.path.join('data', session_name)


def build_condition_code(event, meta: dict = None) -> str:
    """
    Build condition code for filename.
    
    For validation mode with profile_validation_name, uses the full profile name.
    Otherwise uses exactly 3-letter codes (e.g., "INT", "LEF", "SYN").
    
    Args:
        event: ExperimentEvent instance
        meta: Optional meta section
        
    Returns:
        Condition code for filename (profile name or 3-letter code)
    """
    if not meta or not meta.get('mode'):
        # Legacy: return condition label as-is (may be >3 chars for backward compat)
        return event.condition_label
    
    mode = meta['mode']
    
    if mode == 'validation':
        # Check for profile validation mode (uses full profile name)
        profile_name = meta.get('profile_validation_name')
        if profile_name:
            # Sanitize profile name for filesystem
            safe_name = ''.join(c for c in profile_name if c.isalnum() or c in '_-')
            return safe_name if safe_name else "UNK"
        
        # Standard validation scenario code
        scenario = meta.get('validation_scenario', 'unknown')
        base_code = VALIDATION_SCENARIO_CODES.get(scenario, "UNK")
        
        # For known_signal_injection, append CH#_FREQUENCYHZ
        if scenario == 'known_signal_injection':
            injected_signal = meta.get('injected_signal')
            if injected_signal:
                channel = injected_signal.get('channel', 'CH1')
                freq_hz = injected_signal.get('freq_hz', 0)
                return f"{base_code}_{channel}_{freq_hz}HZ"
        
        return base_code
    
    elif mode == 'experiment':
        # Map condition label to 3-letter code
        label = event.condition_label.upper()
        return EXPERIMENT_CONDITION_CODES.get(label, label[:3])
    
    else:
        return event.condition_label


def build_sidecar_metadata(event, config: dict) -> dict:
    """
    Build sidecar JSON metadata with full details.
    
    Args:
        event: ExperimentEvent instance
        config: Full configuration dict
        
    Returns:
        Metadata dictionary for JSON serialization
    """
    meta = config.get('meta', {})
    frames = config['routine']['frames_per_event']
    samples = frames * SAMPLES_PER_FRAME
    duration_seconds = samples / SAMPLE_RATE if SAMPLE_RATE > 0 else 0
    
    # Base metadata (always present)
    sidecar = {
        "mode": meta.get('mode', 'legacy'),
        "timestamp": datetime.now().isoformat(),
        "firmware": meta.get('firmware', 'FW2'),
        "board": meta.get('board', 'R2'),
        "sample_rate_sps": SAMPLE_RATE,
        "frames": frames,
        "samples": samples,
        "duration_seconds": round(duration_seconds, 2),
        "led_enabled": config['routine']['led_enabled'],
        "led_color": event.led_color,
        "event_num": event.event_num
    }
    
    if meta.get('mode') == 'validation':
        scenario = meta.get('validation_scenario')
        sidecar.update({
            "scenario": scenario,
            "scenario_code": VALIDATION_SCENARIO_CODES.get(scenario, "UNK"),
            "subject_id": None,
            "condition": None,
            "injected_signal": meta.get('injected_signal'),  # None or {freq_hz, channel}
            "total_events": 1  # Validation typically runs single events
        })
    
    elif meta.get('mode') == 'experiment':
        sidecar.update({
            "scenario": None,
            "scenario_code": None,
            "subject_id": meta.get('subject_id'),
            "condition": event.condition_label,
            "condition_code": build_condition_code(event, meta),
            "injected_signal": None,
            "total_events": config['experiment']['num_events'],
            "random_seed": config['experiment'].get('random_seed')
        })
    
    else:
        # Legacy mode
        sidecar.update({
            "scenario": None,
            "subject_id": None,
            "condition": event.condition_label,
            "total_events": config['experiment']['num_events'] if config else 1
        })
    
    return sidecar

# ========================================================================


class ExperimentEvent:
    """Handles a single experiment event (streaming + save)"""
    
    def __init__(self, event_num: int, led_color: str, condition_label: str = None, 
                 config: dict = None):
        self.event_num = event_num
        self.led_color = led_color
        self.condition_label = condition_label or led_color  # Fallback to color for legacy mode
        self.config = config  # Configuration dict (for config mode)
        self.frames_received = 0
        self.samples_received = 0
        self.bytes_received = 0
        self.start_time = None
        self.end_time = None
        self.raw_frames = []
        self.last_frame_time = None
        self.success = False
        self.error_message = None
        self.filename = None
        
    def parse_frame_header(self, data):
        """Parse 16-byte frame header"""
        if len(data) < FRAME_HEADER_SIZE:
            return None
        
        t1_first_drdy_us = struct.unpack('<I', data[0:4])[0]
        t2_last_drdy_delta = data[4]
        t3_tx_ready_delta = data[5]
        packet_count = struct.unpack('<H', data[6:8])[0]
        samples_sent = struct.unpack('<I', data[8:12])[0]
        total_samples = struct.unpack('<I', data[12:16])[0]
        
        return {
            'packet_count': packet_count,
            'samples_sent': samples_sent,
            'total_samples': total_samples
        }
    
    async def prepare_routine(self):
        """Prepare routine control for this event"""
        try:
            # Build payload based on mode (config-driven or legacy)
            if self.config:
                # Config mode: Binary LED control via led_enabled
                frames_per_event = self.config['routine']['frames_per_event']
                sound_duration_us = self.config['routine']['sound_duration_us']
                led_enabled = self.config['routine']['led_enabled']
                
                # LED control: send event color OR "LED_OFF"
                led_color = self.led_color if led_enabled else "LED_OFF"
                
                payload = {
                    "num_frames": frames_per_event,
                    "led_color": led_color,
                    "sound_duration_us": sound_duration_us
                }
            else:
                # Legacy mode: use global constants
                payload = {
                    "num_frames": FRAMES_PER_EVENT,
                    "led_color": self.led_color,
                    "sound_duration_us": SOUND_DURATION_US
                }
            
            response = requests.post(
                f"{HTTP_URL}/api/v1/routines/control/prepare",
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                return True
            else:
                self.error_message = f"Prepare failed: HTTP {response.status_code}"
                return False
        except requests.exceptions.RequestException as e:
            self.error_message = f"Prepare HTTP error: {e}"
            return False
    
    async def execute_routine(self):
        """Execute routine (deterministic start)"""
        try:
            response = requests.post(
                f"{HTTP_URL}/api/v1/routines/control/execute",
                timeout=10
            )
            if response.status_code == 200:
                self.start_time = time.time()
                return True
            else:
                self.error_message = f"Execute failed: HTTP {response.status_code}"
                return False
        except requests.exceptions.RequestException as e:
            self.error_message = f"Execute HTTP error: {e}"
            return False
    
    async def receive_frames(
        self,
        websocket,
        experiment_start: float,
        status_cb: callable = None,
        stop_flag: threading.Event = None
    ):
        """Receive frames via WebSocket with debug output"""
        self.last_frame_time = time.time()
        first_frame_time = None
        last_debug_frame = 0
        
        # Determine expected frames (config-driven or legacy)
        expected_frames = self.config['routine']['frames_per_event'] if self.config else FRAMES_PER_EVENT
        
        try:
            while True:
                # Check for timeout
                if time.time() - self.last_frame_time > TIMEOUT_NO_DATA:
                    self.error_message = f"Timeout: no data for {TIMEOUT_NO_DATA}s"
                    break
                
                # Receive with timeout
                try:
                    data = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=0.1
                    )
                    self.last_frame_time = time.time()
                    
                    if first_frame_time is None:
                        first_frame_time = self.last_frame_time
                        
                except asyncio.TimeoutError:
                    continue
                
                # Process binary frame
                if isinstance(data, bytes):
                    frame_size = len(data)
                    
                    if frame_size == FRAME_TARGET_SIZE:
                        header = self.parse_frame_header(data)
                        if header:
                            self.frames_received += 1
                            self.samples_received = header['samples_sent']
                            self.bytes_received += frame_size
                            self.raw_frames.append(data)
                            
                            # Debug output every 200 frames
                            if self.frames_received - last_debug_frame >= 200:
                                elapsed = time.time() - first_frame_time if first_frame_time else 0
                                fps = self.frames_received / elapsed if elapsed > 0 else 0
                                t_offset = time.time() - experiment_start
                                print(f"    [E{self.event_num}] T+{t_offset:.1f}s | Frame {self.frames_received}/{expected_frames} | "
                                      f"{fps:.0f} fps | {self.bytes_received/1024:.0f} KB | pkts={header['packet_count']}")
                                
                                # Send frame progress update with exception handling
                                if status_cb:
                                    try:
                                        status_cb({
                                            "type": "frame_progress",
                                            "event": self.event_num,
                                            "frames_received": self.frames_received,
                                            "expected_frames": expected_frames
                                        })
                                    except Exception:
                                        pass  # Never break acquisition for UI reporting
                                
                                last_debug_frame = self.frames_received
                            
                            # Check if complete
                            if self.frames_received >= expected_frames:
                                self.success = True
                                break
                        
        except websockets.exceptions.ConnectionClosed:
            self.error_message = "WebSocket connection closed"
        except Exception as e:
            self.error_message = f"Receive error: {e}"
        
        self.end_time = time.time()
        
        # Final debug output
        duration = self.end_time - self.start_time if self.start_time else 0
        t_offset = time.time() - experiment_start
        if self.success:
            print(f"    [E{self.event_num}] T+{t_offset:.1f}s | COMPLETE | {self.frames_received} frames in {duration:.2f}s")
        else:
            print(f"    [E{self.event_num}] T+{t_offset:.1f}s | FAILED | {self.frames_received}/{expected_frames} frames | {self.error_message}")
    
    def save_binary(self, output_folder: str) -> bool:
        """
        Save raw binary + sidecar JSON with strict parser-compatible naming.
        
        Filename format: YYYYMMDD_HHMMSS_FW#_R#_XXX_FRAMES_##k.bin
        Where:
        - XXX = exactly 3 uppercase letters
        - FRAMES = frame count (not samples!)
        - ##k = sample rate in kHz (digits only, 'k' appended once)
        
        Returns:
            True if save successful, False otherwise
        """
        if len(self.raw_frames) == 0:
            self.error_message = "No frames to save"
            return False
        
        # Timestamp: YYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        
        # Extract meta
        meta = self.config.get('meta') if self.config else None
        firmware = meta.get('firmware', 'FW2') if meta else 'FW2'
        board = meta.get('board', 'R2') if meta else 'R2'
        
        # FRAMES (not samples!)
        frames = self.config['routine']['frames_per_event'] if self.config else FRAMES_PER_EVENT
        
        # Sample rate: digits only (e.g., "16" for 16kSPS)
        if SAMPLE_RATE >= 1000:
            rate_num = SAMPLE_RATE // 1000  # 16000 → 16
        elif SAMPLE_RATE > 0:
            rate_num = SAMPLE_RATE
        else:
            rate_num = 0
        
        # Build 3-letter condition code
        code = build_condition_code(self, meta)
        
        # Filename: YYYYMMDD_HHMMSS_FW#_R#_XXX_FRAMES_##k.bin
        base_name = f"{timestamp}_{firmware}_{board}_{code}_{frames}_{rate_num}k"
        self.filename = f"{base_name}.bin"
        
        bin_path = os.path.join(output_folder, self.filename)
        json_path = os.path.join(output_folder, f"{base_name}.json")
        
        # Create folder if needed (handles nested paths)
        os.makedirs(output_folder, exist_ok=True)
        
        # Write binary file
        try:
            with open(bin_path, 'wb') as f:
                for frame in self.raw_frames:
                    f.write(frame)
        except Exception as e:
            self.error_message = f"Binary save failed: {e}"
            return False
        
        # Write sidecar JSON (warn if fails, but don't abort)
        try:
            sidecar = build_sidecar_metadata(self, self.config)
            with open(json_path, 'w') as f:
                json.dump(sidecar, f, indent=2)
        except Exception as e:
            # Log warning but continue (binary is more important)
            print(f"    ⚠ Sidecar JSON save failed: {e}")
        
        return True


async def stream_event(
    event: ExperimentEvent,
    experiment_start: float,
    status_cb: callable = None,
    stop_flag: threading.Event = None
) -> ExperimentEvent:
    """
    Stream frames for a single event (runs concurrently via asyncio.create_task).
    This function connects to WebSocket and receives all frames for the event.
    """
    # Small delay for route activation
    await asyncio.sleep(0.3)
    
    # Receive frames
    try:
        async with websockets.connect(WS_URL) as websocket:
            await event.receive_frames(websocket, experiment_start, status_cb, stop_flag)
    except Exception as e:
        event.error_message = f"WebSocket connection error: {e}"
        t_offset = time.time() - experiment_start
        print(f"    [E{event.event_num}] T+{t_offset:.1f}s | WS ERROR: {event.error_message}")
    
    return event


def background_save(event: ExperimentEvent, output_folder: str) -> tuple:
    """Background save function for ThreadPoolExecutor"""
    success = event.save_binary(output_folder)
    return (event.event_num, success, event.filename, event.error_message)


async def run_config_experiment(
    config: dict,
    schedule: list,
    skip_prompts: bool = False,
    status_cb: callable = None,
    stop_flag: threading.Event = None,
    skip_hardware_setup: bool = False
) -> bool:
    """
    Run config-driven experiment with optional hooks for web UI.
    
    Args:
        config: Configuration dictionary
        schedule: List of (led_color, condition_label) tuples
        skip_prompts: If True, skip user confirmation prompts (for web mode)
        status_cb: Optional callback for status updates (for web UI)
        stop_flag: Optional threading.Event to check for stop requests
        skip_hardware_setup: If True, skip profile loading and register dump
                            (useful when caller already handled these steps)
        
    Returns:
        True if successful (including clean stop), False on error
    """
    global HTTP_URL, WS_URL, ESP32_IP
    
    # Set globals from config (required for ExperimentEvent API calls)
    ESP32_IP = config['esp32']['ip']
    HTTP_URL = f"http://{ESP32_IP}"
    WS_URL = f"ws://{ESP32_IP}/ws"
    
    # Extract values from config
    num_events = config['experiment']['num_events']
    inter_event_delay = config['experiment'].get('inter_event_delay', 0.5)
    profile_path = config['experiment'].get('profile_path', None)
    
    # ============ HARDWARE SETUP (Profile Loading + Register Dump) ============
    # These steps ensure the ADS1299 is properly configured before streaming
    if not skip_hardware_setup:
        # Step 1: Load and apply profile if specified
        if profile_path:
            print(f"\n[HARDWARE SETUP] Loading ADS1299 profile...")
            if status_cb:
                try:
                    status_cb({"type": "setup", "step": "profile_loading", "path": profile_path})
                except Exception:
                    pass
            
            if not load_and_apply_profile(profile_path):
                print(f"\n✗ Profile loading failed: {profile_path}")
                if status_cb:
                    try:
                        status_cb({"type": "error", "message": f"Profile loading failed: {profile_path}"})
                    except Exception:
                        pass
                return False
            
            # Brief delay after profile application for hardware to settle
            await asyncio.sleep(0.5)
            print(f"    ✓ Profile applied successfully")
        else:
            print(f"\n[HARDWARE SETUP] No profile specified - using current ADS1299 configuration")
        
        # Step 2: Dump and verify ADS1299 registers
        print(f"\n[HARDWARE SETUP] Verifying ADS1299 configuration...")
        if status_cb:
            try:
                status_cb({"type": "setup", "step": "register_dump"})
            except Exception:
                pass
        
        registers = dump_and_verify_registers()
        if not registers:
            print(f"\n✗ Could not read ADS1299 registers")
            if status_cb:
                try:
                    status_cb({"type": "error", "message": "Could not read ADS1299 registers"})
                except Exception:
                    pass
            return False
        
        # Emit register dump info to status callback
        if status_cb:
            try:
                status_cb({
                    "type": "setup",
                    "step": "register_dump_complete",
                    "sample_rate": SAMPLE_RATE,
                    "registers": registers
                })
            except Exception:
                pass
        
        print(f"    ✓ ADS1299 registers verified (Sample rate: {SAMPLE_RATE} SPS)")
    # =========================================================================
    
    # Compute output folder based on meta section (supports nested folders)
    output_folder = compute_output_folder(config)
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ Output folder ready: {output_folder}/")
    
    # Emit running state
    if status_cb:
        try:
            status_cb({"type": "state", "status": "running"})
        except Exception:
            pass
    
    save_executor = ThreadPoolExecutor(max_workers=MAX_SAVE_WORKERS)
    pending_saves = []
    results = []
    event_durations = []
    experiment_start = time.time()
    
    # Run events from schedule
    for idx, (led_color, condition_label) in enumerate(schedule, 1):
        # Check stop flag ONCE at loop start (canonical location)
        if stop_flag and stop_flag.is_set():
            print(f"\n⚠ Experiment stopped cleanly after {idx-1} events")
            if status_cb:
                try:
                    status_cb({"type": "state", "status": "stopped"})
                except Exception:
                    pass
            # Clean stop = success
            save_executor.shutdown(wait=True)
            return True
        
        event_start = time.time()
        t_offset = event_start - experiment_start
        
        event = ExperimentEvent(idx, led_color, condition_label, config)
        
        print(f"\n{'='*60}")
        print(f"EVENT {idx}/{num_events} | LED: {led_color} | Condition: {condition_label} | T+{t_offset:.1f}s")
        print(f"{'='*60}")
        
        # Emit event start
        if status_cb:
            try:
                status_cb({
                    "type": "event_start",
                    "event": idx,
                    "total": num_events,
                    "led_color": led_color,
                    "condition": condition_label,
                    "expected_frames": config['routine']['frames_per_event']
                })
            except Exception:
                pass
        
        print(f"  [1] Preparing routine...")
        if not await event.prepare_routine():
            print(f"      ✗ {event.error_message}")
            results.append(event)
            if idx < num_events:
                await asyncio.sleep(inter_event_delay)
            continue
        print(f"      ✓ Prepared")
        
        print(f"  [2] Executing routine...")
        if not await event.execute_routine():
            print(f"      ✗ {event.error_message}")
            results.append(event)
            if idx < num_events:
                await asyncio.sleep(inter_event_delay)
            continue
        print(f"      ✓ Started streaming")
        
        print(f"  [3] Receiving frames...")
        await stream_event(event, experiment_start, status_cb, stop_flag)
        
        stream_duration = event.end_time - event.start_time if event.start_time and event.end_time else 0
        event_durations.append(stream_duration)
        
        if event.success:
            expected_frames = config['routine']['frames_per_event']
            print(f"      ✓ Received {event.frames_received}/{expected_frames} frames in {stream_duration:.2f}s")
            
            # Emit event done
            if status_cb:
                try:
                    status_cb({
                        "type": "event_done",
                        "event": idx,
                        "success": True,
                        "error": None
                    })
                except Exception:
                    pass
            
            print(f"  [4] Queuing background save...")
            future = save_executor.submit(background_save, event, output_folder)
            pending_saves.append((event, future))
            print(f"      ✓ Save queued")
        else:
            print(f"      ✗ {event.error_message}")
            # Emit event failed
            if status_cb:
                try:
                    status_cb({
                        "type": "event_done",
                        "event": idx,
                        "success": False,
                        "error": event.error_message
                    })
                except Exception:
                    pass
        
        results.append(event)
        
        if idx < num_events:
            await asyncio.sleep(inter_event_delay)
    
    # Wait for saves
    print("\n" + "="*60)
    print(f"Waiting for {len(pending_saves)} background saves...")
    print("="*60)
    
    save_successes = 0
    save_failures = 0
    saved_files = []
    
    for event, future in pending_saves:
        try:
            event_num, success, filename, error_msg = future.result(timeout=30)
            if success:
                save_successes += 1
                saved_files.append(filename)
                print(f"  ✓ Event {event_num}: {filename}")
                
                # Emit file saved
                if status_cb:
                    try:
                        status_cb({
                            "type": "file_saved",
                            "event": event_num,
                            "filename": filename,
                            "success": True
                        })
                    except Exception:
                        pass
            else:
                save_failures += 1
                print(f"  ✗ Event {event_num}: {error_msg}")
                
                # Emit save failure
                if status_cb:
                    try:
                        status_cb({
                            "type": "file_saved",
                            "event": event_num,
                            "filename": None,
                            "success": False
                        })
                    except Exception:
                        pass
        except Exception as e:
            save_failures += 1
            print(f"  ✗ Event {event.event_num}: {e}")
    
    save_executor.shutdown(wait=True)
    
    # Summary
    successes = sum(1 for e in results if e.success)
    failures = sum(1 for e in results if not e.success)
    total_duration = time.time() - experiment_start
    avg_stream = sum(event_durations) / len(event_durations) if event_durations else 0
    
    print("\n" + "="*70)
    print("=== EXPERIMENT SUMMARY ===")
    print("="*70)
    print(f"\nTotal events:    {num_events}")
    print(f"Successes:       {successes}")
    print(f"Failures:        {failures}")
    print(f"Saved files:     {save_successes}")
    print(f"Total time:      {total_duration:.1f}s")
    print(f"Avg stream time: {avg_stream:.2f}s")
    print(f"Output:          {output_folder}/")
    
    if saved_files:
        print(f"\n📁 Saved files:")
        for f in saved_files:
            print(f"    ✓ {f}")
    
    # Emit summary
    if status_cb:
        try:
            status_cb({
                "type": "summary",
                "stream_successes": successes,
                "stream_failures": failures,
                "save_successes": save_successes,
                "save_failures": save_failures,
                "output_folder": output_folder
            })
        except Exception:
            pass
    
    print("\n" + "="*70)
    if failures == 0 and save_failures == 0:
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
        return True
    else:
        print(f"⚠️ COMPLETED WITH {failures + save_failures} FAILURE(S)")
        return False


async def run_experiment(num_events: int, profile_path: str = None):
    """
    Run the complete experiment with SEQUENTIAL streaming and background saves.
    
    Architecture:
    - Events run SEQUENTIALLY (hardware doesn't support overlapping routines)
    - Each event: prepare → execute → stream → queue background save
    - File saving runs in background ThreadPoolExecutor (non-blocking)
    - Inter-event delay allows hardware to reset between acquisitions
    """
    global HTTP_URL, WS_URL
    
    print("="*70)
    print("=== ADS1299 EXPERIMENT RUNNER (SEQUENTIAL) ===")
    print("="*70)
    print(f"Device:              {ESP32_IP}")
    print(f"Events:              {num_events}")
    print(f"Frames per event:    {FRAMES_PER_EVENT}")
    print(f"Samples per event:   {FRAMES_PER_EVENT * SAMPLES_PER_FRAME}")
    print(f"Sound duration:      {SOUND_DURATION_US / 1000000:.1f} seconds")
    print(f"Inter-event delay:   {INTER_EVENT_DELAY}s (hardware reset)")
    print(f"Output folder:       {OUTPUT_FOLDER}/")
    print(f"Profile:             {profile_path if profile_path else 'default (no change)'}")
    
    # Ensure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"\n✓ Output folder ready: {OUTPUT_FOLDER}/")
    
    # Step 0: Load profile if specified
    if profile_path:
        if not load_and_apply_profile(profile_path):
            print("\n✗ EXPERIMENT ABORTED: Profile loading failed")
            return False
        await asyncio.sleep(0.5)
    
    # Step 1: Dump and verify registers
    print("\n[CONFIG] Verifying ADS1299 configuration...")
    registers = dump_and_verify_registers()
    if not registers:
        print("\n✗ EXPERIMENT ABORTED: Could not read registers")
        return False
    
    # Step 2: Prompt user for confirmation
    print("\n" + "="*60)
    print("Please verify the configuration above is correct.")
    user_input = input("Continue with experiment? [y/N]: ").strip().lower()
    if user_input != 'y':
        print("\n✗ EXPERIMENT ABORTED: User cancelled")
        return False
    
    print("\n" + "="*70)
    print(f"STARTING EXPERIMENT: {num_events} events (sequential)")
    print("  📊 SEQUENTIAL MODE: Complete each stream before starting next")
    print("  💾 BACKGROUND SAVES: File writes don't block next stream")
    print("="*70)
    
    # Create thread pool for background file saving
    save_executor = ThreadPoolExecutor(max_workers=MAX_SAVE_WORKERS)
    pending_saves = []       # List of (event, Future) tuples for saves
    results = []
    event_durations = []     # Track streaming durations
    
    experiment_start = time.time()
    
    # Run events SEQUENTIALLY
    for i in range(1, num_events + 1):
        event_start = time.time()
        t_offset = event_start - experiment_start
        
        # Select random LED color
        led_color = random.choice(VALID_COLORS)
        event = ExperimentEvent(i, led_color)
        
        print(f"\n{'='*60}")
        print(f"EVENT {i}/{num_events} | Color: {led_color} | T+{t_offset:.1f}s")
        print(f"{'='*60}")
        
        # Step 1: Prepare routine
        print(f"  [1] Preparing routine...")
        if not await event.prepare_routine():
            print(f"      ✗ {event.error_message}")
            results.append(event)
            # Still wait before next attempt
            if i < num_events:
                print(f"\n  ⏱️ Waiting {INTER_EVENT_DELAY}s before next event...")
                await asyncio.sleep(INTER_EVENT_DELAY)
            continue
        print(f"      ✓ Prepared")
        
        # Step 2: Execute routine (triggers hardware)
        print(f"  [2] Executing routine...")
        if not await event.execute_routine():
            print(f"      ✗ {event.error_message}")
            results.append(event)
            if i < num_events:
                print(f"\n  ⏱️ Waiting {INTER_EVENT_DELAY}s before next event...")
                await asyncio.sleep(INTER_EVENT_DELAY)
            continue
        print(f"      ✓ Started streaming")
        
        # Step 3: Receive frames (BLOCKING - must complete before next event)
        print(f"  [3] Receiving frames...")
        await stream_event(event, experiment_start)
        
        stream_duration = event.end_time - event.start_time if event.start_time and event.end_time else 0
        event_durations.append(stream_duration)
        
        if event.success:
            print(f"      ✓ Received {event.frames_received}/{FRAMES_PER_EVENT} frames in {stream_duration:.2f}s")
            
            # Step 4: Queue background save (NON-BLOCKING)
            print(f"  [4] Queuing background save...")
            future = save_executor.submit(background_save, event, OUTPUT_FOLDER)
            pending_saves.append((event, future))
            print(f"      ✓ Save queued (running in background)")
        else:
            print(f"      ✗ {event.error_message}")
        
        results.append(event)
        
        # Inter-event delay for hardware reset
        if i < num_events:
            print(f"\n  ⏱️ Hardware reset delay: {INTER_EVENT_DELAY}s")
            await asyncio.sleep(INTER_EVENT_DELAY)
    
    # Wait for all pending saves to complete
    print("\n" + "="*60)
    print(f"All streams complete. Waiting for {len(pending_saves)} background saves...")
    print("="*60)
    
    save_successes = 0
    save_failures = 0
    saved_files = []
    
    for event, future in pending_saves:
        try:
            event_num, success, filename, error_msg = future.result(timeout=30)
            if success:
                save_successes += 1
                saved_files.append(filename)
                print(f"  ✓ Event {event_num}: {filename}")
            else:
                save_failures += 1
                print(f"  ✗ Event {event_num}: {error_msg}")
        except Exception as e:
            save_failures += 1
            print(f"  ✗ Event {event.event_num}: Save error - {e}")
    
    # Shutdown executor
    save_executor.shutdown(wait=True)
    
    # Calculate statistics
    successes = sum(1 for e in results if e.success)
    failures = sum(1 for e in results if not e.success)
    total_duration = time.time() - experiment_start
    avg_stream_duration = sum(event_durations) / len(event_durations) if event_durations else 0
    
    # Print experiment summary
    print("\n")
    print("="*70)
    print("=== EXPERIMENT SUMMARY ===")
    print("="*70)
    print(f"\nTotal events:        {num_events}")
    print(f"Stream successes:    {successes}")
    print(f"Stream failures:     {failures}")
    print(f"Save successes:      {save_successes}")
    print(f"Save failures:       {save_failures}")
    print(f"Total duration:      {total_duration:.1f}s")
    print(f"Avg stream time:     {avg_stream_duration:.2f}s")
    print(f"Avg event interval:  {total_duration / num_events:.2f}s")
    print(f"Output folder:       {OUTPUT_FOLDER}/")
    
    # List successful files
    if saved_files:
        print(f"\n📁 Files saved ({len(saved_files)}):")
        for filename in saved_files:
            print(f"    ✓ {filename}")
    
    # List failures
    if failures > 0 or save_failures > 0:
        print(f"\n❌ Failed events:")
        for event in results:
            if not event.success:
                print(f"    Event {event.event_num}: {event.error_message}")
    
    print("\n" + "="*70)
    
    if failures == 0 and save_failures == 0:
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
    else:
        print(f"⚠️ EXPERIMENT COMPLETED WITH {failures + save_failures} FAILURE(S)")
    
    return failures == 0 and save_failures == 0


if __name__ == "__main__":
    args = parse_arguments()
    
    # Config-first execution: try to load config.json by default
    if args.config:
        config_path = args.config
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
    
    # Declare globals before modifying them
    HTTP_URL = ""
    WS_URL = ""
    ESP32_IP = "node.local"
    
    try:
        # Attempt to load configuration
        print(f"\n🔧 ADS1299 Experiment Runner")
        print(f"=" * 70)
        print(f"\n[STEP 1/5] Loading configuration...")
        config = load_config(config_path)
        config = merge_config_with_args(config, args)
        validate_experiment_config(config)
        
        # Extract config values
        ESP32_IP = config['esp32']['ip']
        HTTP_URL = f"http://{ESP32_IP}"
        WS_URL = f"ws://{ESP32_IP}/ws"
        
        num_events = config['experiment']['num_events']
        profile_path = config['experiment'].get('profile_path', None)
        output_folder = config['experiment'].get('output_folder', 'experiments')
        inter_event_delay = config['experiment'].get('inter_event_delay', 0.5)
        
        # Step 2: Setup reproducibility
        print(f"\n[STEP 2/5] Setting up reproducibility...")
        setup_random_seed(config)
        
        # Step 3: Generate event schedule
        print(f"\n[STEP 3/5] Generating event schedule...")
        schedule = generate_event_schedule(
            num_events,
            config['conditions']['mapping'],
            config['conditions']['enforce_equal_condition_count']
        )
        display_event_schedule(
            schedule,
            config['conditions']['enforce_equal_condition_count'],
            config['experiment'].get('random_seed')
        )
        
        # Step 4: Load profile if specified
        if profile_path:
            print(f"\n[STEP 4/5] Loading ADS1299 profile...")
            if not load_and_apply_profile(profile_path):
                print("\n✗ EXPERIMENT ABORTED: Profile loading failed")
                sys.exit(1)
            asyncio.run(asyncio.sleep(0.5))
        else:
            print(f"\n[STEP 4/5] Skipping profile load (none specified)")
        
        # Step 5: Verify registers and get user confirmation
        print(f"\n[STEP 5/5] Verifying ADS1299 configuration...")
        registers = dump_and_verify_registers()
        if not registers:
            print("\n✗ EXPERIMENT ABORTED: Could not read registers")
            sys.exit(1)
        
        print("\n" + "="*70)
        print("Please review the configuration, event schedule, and register dump above.")
        user_input = input("Continue with experiment? [y/N]: ").strip().lower()
        if user_input != 'y':
            print("\n✗ EXPERIMENT ABORTED: User cancelled")
            sys.exit(0)
        
        # Run config-driven experiment (call module-level function)
        print(f"\n" + "="*70)
        print(f"STARTING CONFIG-DRIVEN EXPERIMENT")
        print(f"=" * 70)
        print(f"📋 Configuration: {config_path}")
        print(f"🎲 Random seed: {config['experiment'].get('random_seed', 'none')}")
        print(f"⚖️  Equal counts: {config['conditions']['enforce_equal_condition_count']}")
        print(f"🎯 Events: {num_events}")
        print(f"=" * 70)
        
        # Call the module-level run_config_experiment function
        # skip_hardware_setup=True because Steps 4/5 already handled profile loading and register dump
        success = asyncio.run(run_config_experiment(
            config=config,
            schedule=schedule,
            skip_prompts=False,       # CLI mode: keep prompts
            status_cb=None,           # No callbacks in CLI
            stop_flag=None,           # No stop flag in CLI
            skip_hardware_setup=True  # Already done in Steps 4/5 above
        ))
        sys.exit(0 if success else 1)
        
    except ValueError as e:
        # Config file issue
        if args.config:
            # User explicitly provided config - fail
            print(f"\n✗ CONFIG ERROR: {e}")
            sys.exit(1)
        else:
            # No explicit --config, try legacy fallback
            if args.events is None:
                print(f"\n✗ No config.json found and no --events specified")
                print(f"   Create config.json or use: python run_experiment.py --events NUM")
                sys.exit(1)
            
            # Fall back to legacy mode
            ESP32_IP = args.ip
            HTTP_URL = f"http://{ESP32_IP}"
            WS_URL = f"ws://{ESP32_IP}/ws"
            
            print(f"\n🔧 ADS1299 Experiment Runner (Legacy Mode)")
            print(f"  --events: {args.events}")
            print(f"  --ip: {ESP32_IP}")
            
            try:
                success = asyncio.run(run_experiment(args.events, args.profile))
                sys.exit(0 if success else 1)
            except KeyboardInterrupt:
                print("\n\n✗ Interrupted")
                sys.exit(1)
            except Exception as e:
                print(f"\n✗ Error: {e}")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n✗ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
