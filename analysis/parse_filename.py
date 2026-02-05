"""
Filename Metadata Parser

Parse metadata from ADS1299 binary filenames.
Supports multiple filename formats: KSI, NEW, OLD, and Functional.
"""

import re
from pathlib import Path


def _format_date(date_str: str) -> str:
    """Convert YYMMDD or YYYYMMDD to YYYY-MM-DD format."""
    if len(date_str) == 8:  # YYYYMMDD
        return f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
    elif len(date_str) == 6:  # YYMMDD → 20YY-MM-DD
        return f"20{date_str[0:2]}-{date_str[2:4]}-{date_str[4:6]}"
    return date_str  # Fallback


def _format_time(time_str: str) -> str:
    """Convert HHMM or HHMMSS to HH:MM or HH:MM:SS format."""
    if len(time_str) == 4:  # HHMM
        return f"{time_str[0:2]}:{time_str[2:4]}"
    elif len(time_str) == 6:  # HHMMSS
        return f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
    return time_str  # Fallback


def detect_test_type(filename_meta: dict) -> str:
    """Auto-detect test type from filename pattern."""
    condition = filename_meta.get('condition', '').upper()
    
    if condition.startswith('FUN'):
        return "functional"   # FUNEC, FUNEB, FUNEO
    elif condition == 'KSI':
        return "injection"    # Signal injection
    else:
        return "hardware"     # INT/EXT/SYN etc.


def parse_filename_metadata(bin_path: Path) -> dict:
    """
    Parse metadata from filename.
    
    Supports formats:
    - NEW:  YYMMDD_HHMMSS_FWX_RY_COND_NNNN_NNk.bin
    - NEW:  YYYYMMDD_HHMMSS_FWX_RY_COND_NNNN_NNk.bin
    - KSI:  YYMMDD_HHMMSS_FWX_RY_KSI_CHX_XXHZ_NNNN_NNk.bin
    - FUN:  YYMMDD_HHMMSS_FWX_RY_FUNEC_NNNN_NNk.bin
    - OLD:  DD-MM-FWX_VY_NNNNframes_NNkhz.bin
    
    Parameters
    ----------
    bin_path : Path
        Path to the .bin file
    
    Returns
    -------
    dict
        Metadata dictionary with keys:
        - date, time, firmware, board, condition
        - n_frames_declared, sampling_rate_khz
        - format, display_name
        - target_channel, injection_freq_hz (for KSI only)
    """
    filename = bin_path.stem
    
    meta = {
        "date": None,
        "time": None,
        "firmware": None,
        "board": None,
        "condition": None,
        "n_frames_declared": None,
        "sampling_rate_khz": None,
        "format": None,
        "target_channel": None,
        "injection_freq_hz": None,
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIORITY 1: KSI format (Signal Injection) - Most specific
    # Example: 260130_142403_FW2_R2_KSI_CH4_40HZ_1600_16k.bin
    # ═══════════════════════════════════════════════════════════════════════
    ksi_pattern = r'^(\d{6,8})_(\d{4,6})_FW(\d+)_R(\d+)_KSI_CH(\d+)_(\d+)HZ_(\d+)_(\d+)k'
    ksi_match = re.match(ksi_pattern, filename, re.IGNORECASE)

    if ksi_match:
        meta["date"] = _format_date(ksi_match.group(1))
        meta["time"] = _format_time(ksi_match.group(2))
        meta["firmware"] = f"FW{ksi_match.group(3)}"
        meta["board"] = f"R{ksi_match.group(4)}"
        meta["condition"] = "KSI"
        meta["target_channel"] = int(ksi_match.group(5))
        meta["injection_freq_hz"] = int(ksi_match.group(6))
        meta["n_frames_declared"] = int(ksi_match.group(7))
        meta["sampling_rate_khz"] = int(ksi_match.group(8))
        meta["format"] = "ksi"
        meta["display_name"] = (
            f"{meta['firmware']} | {meta['board']} "
            f"[KSI-CH{meta['target_channel']}@{meta['injection_freq_hz']}Hz] | "
            f"{meta['date']} {meta['time']}"
        )
        return meta
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIORITY 2: NEW format (General + Functional)
    # Examples: 
    #   260130_140604_FW2_R2_INT_1600_16k.bin
    #   260130_151322_FW2_R2_FUNEC_1600_16k.bin
    #   20260130_111927_FW2_R2_INT_1600_16k.bin
    # ═══════════════════════════════════════════════════════════════════════
    new_pattern = r'^(\d{6,8})_(\d{4,6})_FW(\d+)_R(\d+)_?([A-Z]+(?:EC|EB|EO)?)?_?(\d+)_(\d+)k'
    new_match = re.match(new_pattern, filename, re.IGNORECASE)

    if new_match:
        date_str = new_match.group(1)
        time_str = new_match.group(2)
        condition = new_match.group(5)
        
        meta["date"] = _format_date(date_str)
        meta["time"] = _format_time(time_str)
        meta["firmware"] = f"FW{new_match.group(3)}"
        meta["board"] = f"R{new_match.group(4)}"
        meta["condition"] = condition if condition else "General"
        meta["n_frames_declared"] = int(new_match.group(6))
        meta["sampling_rate_khz"] = int(new_match.group(7))
        meta["format"] = "new"
        
        # Build display name
        if condition and condition.upper().startswith('FUN'):
            cond_str = f" [FUNCTIONAL-{condition.upper()}]"
        elif condition:
            cond_str = f" [{condition.upper()}]"
        else:
            cond_str = ""
        
        meta["display_name"] = (
            f"{meta['firmware']} | {meta['board']}{cond_str} | "
            f"{meta['date']} {meta['time']}"
        )
        return meta
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIORITY 3: OLD format
    # Example: 26-01-FW2_V1_1234frames_16khz.bin
    # ═══════════════════════════════════════════════════════════════════════
    old_pattern = r'^(\d{2}-\d{2}).*FW(\d+).*V(\d+).*?(\d+)frames.*?(\d+)khz'
    old_match = re.search(old_pattern, filename, re.IGNORECASE)
    
    if old_match:
        meta["date"] = old_match.group(1)
        meta["firmware"] = f"FW{old_match.group(2)}"
        meta["board"] = f"V{old_match.group(3)}"
        meta["condition"] = "General"
        meta["n_frames_declared"] = int(old_match.group(4))
        meta["sampling_rate_khz"] = int(old_match.group(5))
        meta["format"] = "old"
        meta["display_name"] = f"{meta['firmware']} | {meta['board']} | {meta['date']}"
        return meta
    
    # ═══════════════════════════════════════════════════════════════════════
    # FALLBACK: Unknown format
    # ═══════════════════════════════════════════════════════════════════════
    meta["format"] = "unknown"
    meta["condition"] = "Unknown"
    meta["display_name"] = filename
    return meta
