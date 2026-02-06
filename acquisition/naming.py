"""
Filename, output folder, and sidecar metadata contracts for acquisition system.

Owns the naming conventions and metadata schemas for:
- Output folder structure
- Binary filenames
- JSON sidecar metadata

Functions:
- compute_output_folder() - Build output path from config
- build_condition_code() - Generate codes for filenames
- build_sidecar_metadata() - Build JSON metadata dict
- build_filename() - Build complete filename from event data
"""

import os
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from .constants import (
    SAMPLES_PER_FRAME,
    VALIDATION_SCENARIO_CODES,
    EXPERIMENT_CONDITION_CODES
)

# Type hint for ExperimentEvent without circular import
if TYPE_CHECKING:
    from .streaming import ExperimentEvent


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


def build_condition_code(
    led_color: str,
    condition_label: str,
    meta: Optional[dict] = None
) -> str:
    """
    Build condition code for filename.
    
    For validation mode with profile_validation_name, uses the full profile name.
    Otherwise uses exactly 3-letter codes (e.g., "INT", "LEF", "SYN").
    
    Args:
        led_color: LED color for the event
        condition_label: Condition label from schedule
        meta: Optional meta section from config
        
    Returns:
        Condition code for filename (profile name or 3-letter code)
    """
    if not meta or not meta.get('mode'):
        # Legacy: return condition label as-is (may be >3 chars for backward compat)
        return condition_label
    
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
        label = condition_label.upper()
        return EXPERIMENT_CONDITION_CODES.get(label, label[:3])
    
    else:
        return condition_label


def build_sidecar_metadata(
    event_num: int,
    led_color: str,
    condition_label: str,
    frames_received: int,
    sample_rate: int,
    config: dict
) -> dict:
    """
    Build sidecar JSON metadata with full details.
    
    Args:
        event_num: Event number (1-indexed)
        led_color: LED color used
        condition_label: Condition label from schedule
        frames_received: Actual frames received
        sample_rate: Sample rate in Hz
        config: Full configuration dict
        
    Returns:
        Metadata dictionary for JSON serialization
    """
    meta = config.get('meta', {})
    frames = config['routine']['frames_per_event']
    samples = frames * SAMPLES_PER_FRAME
    duration_seconds = samples / sample_rate if sample_rate > 0 else 0
    
    # Base metadata (always present)
    sidecar = {
        "mode": meta.get('mode', 'legacy'),
        "timestamp": datetime.now().isoformat(),
        "firmware": meta.get('firmware', 'FW2'),
        "board": meta.get('board', 'R2'),
        "sample_rate_sps": sample_rate,
        "frames": frames,
        "frames_received": frames_received,
        "samples": samples,
        "duration_seconds": round(duration_seconds, 2),
        "led_enabled": config['routine']['led_enabled'],
        "led_color": led_color,
        "event_num": event_num
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
            "condition": condition_label,
            "condition_code": build_condition_code(led_color, condition_label, meta),
            "injected_signal": None,
            "total_events": config['experiment']['num_events'],
            "random_seed": config['experiment'].get('random_seed')
        })
    
    else:
        # Legacy mode
        sidecar.update({
            "scenario": None,
            "subject_id": None,
            "condition": condition_label,
            "total_events": config['experiment']['num_events'] if config else 1
        })
    
    return sidecar


def build_filename(
    led_color: str,
    condition_label: str,
    frames: int,
    sample_rate: int,
    meta: Optional[dict] = None
) -> str:
    """
    Build filename for binary data file.
    
    Format: YYYYMMDD_HHMMSS_FW#_R#_XXX_FRAMES_##k.bin
    
    Args:
        led_color: LED color for the event
        condition_label: Condition label from schedule
        frames: Number of frames acquired
        sample_rate: Sample rate in Hz
        meta: Optional meta section from config
        
    Returns:
        Base filename (without path)
    """
    # Timestamp: YYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    
    # Firmware and board version
    firmware = meta.get('firmware', 'FW2') if meta else 'FW2'
    board = meta.get('board', 'R2') if meta else 'R2'
    
    # Sample rate: digits only (e.g., "16" for 16kSPS)
    if sample_rate >= 1000:
        rate_num = sample_rate // 1000
    elif sample_rate > 0:
        rate_num = sample_rate
    else:
        rate_num = 0
    
    # Build condition code
    code = build_condition_code(led_color, condition_label, meta)
    
    # Filename: YYYYMMDD_HHMMSS_FW#_R#_XXX_FRAMES_##k.bin
    base_name = f"{timestamp}_{firmware}_{board}_{code}_{frames}_{rate_num}k"
    
    return base_name
