"""
Timing Integrity Analysis

Δt1 metrics, dropped frame estimation, duplicate detection, frame reordering.
"""

import numpy as np
from .decode_bin import PACKETS_PER_FRAME


# ═══════════════════════════════════════════════════════════════════════════════
# UINT32 UNWRAP
# ═══════════════════════════════════════════════════════════════════════════════

def unwrap_u32_to_i64(x_u32: np.ndarray) -> np.ndarray:
    """
    Standard uint32 unwrap to int64 (adds 2^32 on wrap).
    Does NOT smooth steps; anomalies remain visible in Δt1.
    
    Parameters
    ----------
    x_u32 : np.ndarray
        Array of uint32 values (e.g., t1 timestamps)
    
    Returns
    -------
    np.ndarray
        Unwrapped int64 values
    """
    x = np.asarray(x_u32, dtype=np.uint32).astype(np.uint64)
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64)
    d = np.diff(x.astype(np.int64))
    WRAP = 1 << 32
    d_corr = d.copy()
    d_corr[d < -(WRAP // 2)] += WRAP
    out = np.empty_like(x, dtype=np.int64)
    out[0] = int(x[0])
    out[1:] = out[0] + np.cumsum(d_corr.astype(np.int64))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLES_SENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def decode_samples_sent_to_samples(samples_sent_u32: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Detect whether samples_sent is raw samples (Δ≈50) or Q16.16 (Δ≈50<<16).
    
    Returns
    -------
    samples : np.ndarray
        Decoded sample counts (int64)
    format_str : str
        "raw" or "q16.16"
    """
    ss = unwrap_u32_to_i64(samples_sent_u32.astype(np.uint32))
    if ss.size < 2:
        return ss.astype(np.int64), "raw"

    d = np.diff(ss)
    step_raw = PACKETS_PER_FRAME
    step_q16 = PACKETS_PER_FRAME << 16

    raw_hits = int(np.sum(d == step_raw))
    q16_hits = int(np.sum(d == step_q16))

    if q16_hits > raw_hits:
        return (ss >> 16).astype(np.int64), "q16.16"
    return ss.astype(np.int64), "raw"


def samples_sent_step_stats(meta: dict) -> dict:
    """
    Analyze samples_sent step patterns.
    
    Parameters
    ----------
    meta : dict
        Metadata from binary parser (contains hdr dict)
    
    Returns
    -------
    dict
        Step statistics
    """
    hdr = meta["hdr"]
    ss_samples, ss_fmt = decode_samples_sent_to_samples(hdr["samples_sent"])
    dss = np.diff(ss_samples) if ss_samples.size > 1 else np.array([], dtype=np.int64)

    eq50 = int(np.sum(dss == PACKETS_PER_FRAME))
    mult50 = int(np.sum((dss % PACKETS_PER_FRAME) == 0)) if dss.size else 0
    zeros = int(np.sum(dss == 0))
    neg = int(np.sum(dss < 0))

    # missing estimate from forward jumps that are multiples of 50
    jumps = dss[dss > PACKETS_PER_FRAME]
    jumps_mult = jumps[(jumps % PACKETS_PER_FRAME) == 0]
    missing_est = int(np.sum((jumps_mult // PACKETS_PER_FRAME) - 1)) if jumps_mult.size else 0

    out = {
        "format": ss_fmt,
        "n_transitions": int(dss.size),
        "eq50": eq50,
        "multiple_of_50": mult50,
        "zeros": zeros,
        "negatives": neg,
        "missing_frames_est_from_forward_jumps": missing_est,
        "unique_first10": np.unique(dss)[:10].tolist() if dss.size else [],
    }
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Δt1 TIMING METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def dt1_metrics(meta: dict, *, fs_hz: int, jitter_tol_us: int = 200) -> dict:
    """
    Core frame timing metrics from t1 timestamps.
    
    Parameters
    ----------
    meta : dict
        Metadata with hdr dict
    fs_hz : int
        Sampling rate in Hz
    jitter_tol_us : int
        Tolerance for jitter classification
    
    Returns
    -------
    dict
        Timing metrics including dt1 array
    """
    hdr = meta["hdr"]
    t1_u = unwrap_u32_to_i64(hdr["t1_first_drdy_us"])
    dt1 = np.diff(t1_u) if t1_u.size > 1 else np.array([], dtype=np.int64)

    expected_dt1_us = int(round(PACKETS_PER_FRAME * 1e6 / fs_hz))
    lo = expected_dt1_us - int(jitter_tol_us)
    hi = expected_dt1_us + int(jitter_tol_us)

    outliers = np.where((dt1 < lo) | (dt1 > hi))[0] if dt1.size else np.array([], dtype=int)

    # drop estimate from dt1 multiples (ONLY when close to multiples)
    if dt1.size:
        k = np.rint(dt1 / expected_dt1_us).astype(int)
        k[k < 0] = 0
        tol_us = int(round(0.25 * expected_dt1_us))
        valid = (k >= 1) & (np.abs(dt1 - k * expected_dt1_us) <= tol_us)
        drops_est = np.zeros_like(k)
        drops_est[valid] = np.maximum(k[valid] - 1, 0)
        drops_total = int(np.sum(drops_est))
        valid_frac = float(np.mean(valid))
    else:
        drops_total = 0
        valid_frac = 0.0

    # cadence estimate from median dt1 of positive values
    if dt1.size:
        pos = dt1[dt1 > 0]
        med = float(np.median(pos)) if pos.size else float("nan")
        fs_est = 1e6 * PACKETS_PER_FRAME / med if (pos.size and med > 0) else float("nan")
    else:
        med = float("nan")
        fs_est = float("nan")

    return {
        "expected_dt1_us": expected_dt1_us,
        "lo": lo,
        "hi": hi,
        "t1_first_us": int(t1_u[0]) if t1_u.size else None,
        "t1_last_us": int(t1_u[-1]) if t1_u.size else None,
        "t1_duration_s": float((t1_u[-1] - t1_u[0]) / 1e6) if t1_u.size else None,
        "dt1_min": int(dt1.min()) if dt1.size else None,
        "dt1_mean": float(np.mean(dt1)) if dt1.size else None,
        "dt1_max": int(dt1.max()) if dt1.size else None,
        "dt1_outliers_count": int(outliers.size),
        "dt1_outliers_first_index": int(outliers[0]) if outliers.size else None,
        "dt1_median_us": med,
        "fs_est_hz_from_median_dt1": fs_est,
        "drops_est_from_dt1_multiples": drops_total,
        "dt1_multiple_valid_fraction": valid_frac,
        "dt1": dt1,
        "t1_unwrapped": t1_u,
    }


def estimate_dropped_samples_1sd(meta: dict, *, fs_hz: int) -> dict:
    """
    Estimate dropped samples using 1 SD threshold.
    
    Should be called on SORTED meta (after reorder_by_t1).
    Analyzes all Δt1 values to find outliers indicating dropped frames.
    
    Parameters
    ----------
    meta : dict
        Metadata (should be sorted)
    fs_hz : int
        Sampling rate in Hz
    
    Returns
    -------
    dict
        Dropped sample estimates
    """
    hdr = meta["hdr"]
    t1 = unwrap_u32_to_i64(hdr["t1_first_drdy_us"])
    dt1 = np.diff(t1)
    
    if dt1.size == 0:
        return {
            "error": "No frame transitions",
            "mean_dt1_us": 0,
            "std_dt1_us": 0,
            "threshold_us": 0,
            "outlier_count": 0,
            "total_dropped_samples_est": 0,
            "dropped_per_frame": [],
        }
    
    # Calculate stats from ALL dt1 (sorted data should have no negatives)
    mean_dt1 = float(np.mean(dt1))
    std_dt1 = float(np.std(dt1))
    threshold = mean_dt1 + std_dt1  # 1 SD above mean
    
    # Expected interval
    expected_dt1_us = 1e6 * PACKETS_PER_FRAME / fs_hz  # e.g., 3125 µs @ 16kHz
    us_per_sample = 1e6 / fs_hz  # e.g., 62.5 µs @ 16kHz
    
    # Find outliers (Δt1 > threshold)
    outliers = dt1 > threshold
    outlier_indices = np.where(outliers)[0]
    
    # Estimate dropped samples for each outlier
    dropped_per_frame = []
    for idx in outlier_indices:
        excess_time = dt1[idx] - expected_dt1_us
        # Convert excess time to number of samples
        dropped_samples = int(round(excess_time / us_per_sample))
        dropped_per_frame.append({
            "frame_index": int(idx),
            "dt1_us": int(dt1[idx]),
            "excess_us": int(excess_time),
            "dropped_samples_est": dropped_samples,
        })
    
    total_dropped = sum(d["dropped_samples_est"] for d in dropped_per_frame)
    
    return {
        "mean_dt1_us": mean_dt1,
        "std_dt1_us": std_dt1,
        "threshold_us": threshold,
        "outlier_count": len(outlier_indices),
        "outlier_indices": outlier_indices.tolist()[:20],  # First 20
        "total_dropped_samples_est": total_dropped,
        "dropped_per_frame": dropped_per_frame,
        "method": "1_SD_above_mean",
        "min_dt1_us": int(np.min(dt1)),
        "max_dt1_us": int(np.max(dt1)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COUNTER ROLLOVER CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_counter_near_rollover(meta: dict) -> dict:
    """Check if t1 was near uint32 max during measurement."""
    t1_raw = meta["hdr"]["t1_first_drdy_us"]
    UINT32_MAX = 0xFFFFFFFF
    NEAR_END = UINT32_MAX * 0.95
    
    near_end_count = np.sum(t1_raw > NEAR_END)
    has_low = np.any(t1_raw < 1_000_000)
    rolled_over = (near_end_count > 0) and has_low
    
    return {
        "near_end": near_end_count > 0,
        "likely_rolled_over": rolled_over,
        "max_t1": int(t1_raw.max()),
        "safe_to_sort": not rolled_over,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME REORDERING
# ═══════════════════════════════════════════════════════════════════════════════

def reorder_by_t1(counts_all: np.ndarray, meta: dict) -> tuple:
    """
    Reorder frames chronologically by t1 timestamp.
    
    Parameters
    ----------
    counts_all : np.ndarray
        Shape (8, n_samples) - raw ADC counts
    meta : dict
        Metadata with hdr dict
    
    Returns
    -------
    counts_sorted : np.ndarray
        Reordered counts
    meta_sorted : dict
        Reordered metadata with was_reordered flag
    """
    t1_unwrapped = unwrap_u32_to_i64(meta["hdr"]["t1_first_drdy_us"])
    sort_idx = np.argsort(t1_unwrapped)
    
    # Reorder samples (50 per frame)
    n_frames = len(sort_idx)
    counts_sorted = np.zeros_like(counts_all)
    for new_i, old_i in enumerate(sort_idx):
        counts_sorted[:, new_i*50:(new_i+1)*50] = counts_all[:, old_i*50:(old_i+1)*50]
    
    # Reorder headers
    meta_sorted = meta.copy()
    meta_sorted["hdr"] = {k: v[sort_idx] for k, v in meta["hdr"].items()}
    
    # Store unwrapped t1 to prevent double-unwrapping downstream
    meta_sorted["hdr"]["t1_unwrapped_us"] = t1_unwrapped[sort_idx]
    meta_sorted["was_reordered"] = True
    
    return counts_sorted, meta_sorted


# ═══════════════════════════════════════════════════════════════════════════════
# DUPLICATE/TIMESTAMP VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_timestamps(meta: dict, *, tolerance_us: int = 0) -> dict:
    """
    STRICT timestamp validation - flags ANY duplicate timestamps.
    
    Parameters
    ----------
    meta : dict
        Metadata with hdr dict
    tolerance_us : int
        Maximum allowed Δt1. Default 0 means EXACT duplicates flagged.
        
    Returns
    -------
    dict
        Validation results with severity assessment
    """
    t1 = meta["hdr"]["t1_first_drdy_us"]
    t1_unwrapped = unwrap_u32_to_i64(t1)
    dt1 = np.diff(t1_unwrapped)
    
    if dt1.size == 0:
        return {
            "pass": True,
            "duplicate_count": 0,
            "duplicate_pct": 0.0,
            "severity": "OK",
            "message": "No frame transitions to check",
            "zero_count": 0,
            "negative_count": 0,
        }
    
    # Count duplicates (Δt1 <= tolerance)
    duplicates = int(np.sum(dt1 <= tolerance_us))
    duplicate_pct = 100.0 * duplicates / dt1.size
    
    # Severity assessment
    if duplicates == 0:
        severity = "OK"
        message = "All timestamps unique"
    elif duplicate_pct < 1.0:
        severity = "WARNING"
        message = f"{duplicates} duplicate timestamps (<1%)"
    elif duplicate_pct < 10.0:
        severity = "ERROR"
        message = f"{duplicates} duplicate timestamps ({duplicate_pct:.1f}%)"
    else:
        severity = "CRITICAL"
        message = f"{duplicates} duplicate timestamps ({duplicate_pct:.1f}%) - FIRMWARE BUG!"
    
    return {
        "pass": duplicates == 0,
        "duplicate_count": duplicates,
        "duplicate_pct": duplicate_pct,
        "severity": severity,
        "message": message,
        "zero_count": int(np.sum(dt1 == 0)),  # Exact zeros
        "negative_count": int(np.sum(dt1 < 0)),  # Backward jumps
    }


def validate_t3_t2_physics(meta: dict) -> dict:
    """
    Check for physical impossibility: t3 > t2
    
    t2 = time between last 2 DRDY pulses (ADC sampling interval)
    t3 = time from last DRDY to TX_READY (firmware processing time)
    
    Physical constraint: Firmware cannot be ready (t3) before 
    the ADC finishes sampling (t2). Therefore t3 must be <= t2.
    
    Returns
    -------
    dict
        Validation results
    """
    hdr = meta["hdr"]
    t2_us = hdr["t2_last_drdy_delta_4us"].astype(np.int64) * 4
    t3_us = hdr["t3_tx_ready_delta_4us"].astype(np.int64) * 4
    
    violations = t3_us > t2_us
    violation_count = int(np.sum(violations))
    violation_indices = np.where(violations)[0]
    
    return {
        "pass": violation_count == 0,
        "violation_count": violation_count,
        "violation_pct": 100.0 * violation_count / len(t3_us) if len(t3_us) else 0.0,
        "violation_frames": violation_indices[:10].tolist() if violation_count > 0 else [],
        "severity": "OK" if violation_count == 0 else "CRITICAL",
        "message": "No violations" if violation_count == 0 else f"{violation_count} frames violate physics (t3>t2)"
    }


def analyze_duplicate_frame_data(counts_all: np.ndarray, meta: dict) -> dict:
    """
    For frames with duplicate timestamps (Δt1 = 0), check if the 
    actual channel data is identical or different.
    
    This distinguishes between:
    - Frame retransmission (CRITICAL - data loss)
    - Frozen timestamp counter (ERROR - timing corrupted but data valid)
    
    Parameters
    ----------
    counts_all : np.ndarray
        Shape (8, n_samples) - all channel data
    meta : dict
        Parsed metadata with header info
    
    Returns
    -------
    dict
        Analysis results with severity
    """
    hdr = meta["hdr"]
    t1 = unwrap_u32_to_i64(hdr["t1_first_drdy_us"])
    dt1 = np.diff(t1)
    
    # Find all duplicate timestamp locations (Δt1 = 0)
    dup_indices = np.where(dt1 == 0)[0]
    
    if len(dup_indices) == 0:
        return {
            "has_duplicates": False,
            "duplicate_count": 0,
            "identical_data_count": 0,
            "different_data_count": 0,
            "severity": "OK",
            "message": "No duplicate timestamps found",
        }
    
    # For each duplicate, compare the actual frame data
    identical_count = 0
    different_count = 0
    
    for idx in dup_indices:
        # idx is the transition index
        # Compare frames at positions idx and idx+1
        frame_i = idx
        frame_i_next = idx + 1
        
        # Extract data for these frames (50 samples per frame, 8 channels)
        # Frame data spans columns [frame*50 : (frame+1)*50]
        start_i = frame_i * 50
        end_i = (frame_i + 1) * 50
        start_next = frame_i_next * 50
        end_next = (frame_i_next + 1) * 50
        
        data_i = counts_all[:, start_i:end_i]
        data_i_next = counts_all[:, start_next:end_next]
        
        # Check if the two frames are identical
        is_identical = np.array_equal(data_i, data_i_next)
        
        if is_identical:
            identical_count += 1
        else:
            different_count += 1
    
    # Determine severity based on results
    if identical_count > 0:
        severity = "CRITICAL"
        message = f"{identical_count} duplicate frames (retransmission - DATA LOSS!)"
    elif different_count > 0:
        severity = "ERROR"
        message = f"{different_count} frozen timestamps (counter bug, data valid)"
    else:
        severity = "OK"
        message = "No issues"
    
    return {
        "has_duplicates": True,
        "duplicate_count": len(dup_indices),
        "identical_data_count": identical_count,
        "different_data_count": different_count,
        "duplicate_indices": dup_indices.tolist()[:10],  # First 10 for debugging
        "severity": severity,
        "message": message,
    }
