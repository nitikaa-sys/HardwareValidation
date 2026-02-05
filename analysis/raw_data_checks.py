"""
Raw Data Quality Checks

Status byte decoding, counter analysis, zero/saturation checks.
These checks are performed BEFORE any sorting or processing.
"""

import numpy as np
from .decode_bin import (
    FRAME_BYTES, HEADER_BYTES, PACKETS_PER_FRAME, 
    PACKET_BYTES, STATUS_BYTES
)

# ═══════════════════════════════════════════════════════════════════════════════
# ADC SATURATION THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

FS_POS = (1 << 23) - 1      # +8388607
FS_NEG = -(1 << 23)         # -8388608


# ═══════════════════════════════════════════════════════════════════════════════
# PACKET PAD AND STATUS EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_packet_pad_and_status(bin_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract padding byte and status bytes from all packets.
    
    Parameters
    ----------
    bin_bytes : bytes
        Raw binary file contents
    
    Returns
    -------
    pad : np.ndarray
        Shape (n_frames, 50) - padding bytes per packet
    status : np.ndarray
        Shape (n_frames, 50, 3) - status bytes per packet
    """
    data = np.frombuffer(bin_bytes, dtype=np.uint8)
    if data.size % FRAME_BYTES != 0:
        raise ValueError(f"Bin length {data.size} not divisible by FRAME_BYTES={FRAME_BYTES}.")

    n_frames = data.size // FRAME_BYTES
    frames = data.reshape(n_frames, FRAME_BYTES)
    payload = frames[:, HEADER_BYTES:]                                # (n_frames, 1400)
    packets = payload.reshape(n_frames, PACKETS_PER_FRAME, PACKET_BYTES)  # (n_frames, 50, 28)

    pad = packets[:, :, 27].astype(np.uint8)                          # padding byte per packet
    status = packets[:, :, :STATUS_BYTES].astype(np.uint8)            # first 3 bytes
    return pad, status


def quick_counter_report(pad: np.ndarray, status: np.ndarray, *, n_preview_frames: int = 3) -> dict:
    """
    Analyze padding byte patterns (potential frame/packet counter).
    
    Parameters
    ----------
    pad : np.ndarray
        Shape (n_frames, 50) - padding bytes
    status : np.ndarray
        Shape (n_frames, 50, 3) - status bytes
    n_preview_frames : int
        Number of frames to include in preview
    
    Returns
    -------
    dict
        Counter statistics for logging
    """
    n_frames = int(pad.shape[0])

    pkt_idx = np.arange(PACKETS_PER_FRAME, dtype=np.uint8)
    pct_eq_pktidx = 100.0 * float(np.mean(pad == pkt_idx[None, :]))

    cand0 = pad[:, 0].astype(np.int64)
    candL = pad[:, -1].astype(np.int64)

    def _scores(x: np.ndarray) -> tuple[float, float]:
        d = np.diff(x)
        return float(np.mean(d == 1)), float(np.mean(d >= 0))

    s0_inc1, s0_nondec = _scores(cand0) if cand0.size > 1 else (0.0, 0.0)
    sL_inc1, sL_nondec = _scores(candL) if candL.size > 1 else (0.0, 0.0)

    preview = []
    for i in range(min(n_preview_frames, n_frames)):
        preview.append({
            "frame": i,
            "pad_first12": pad[i, :12].tolist(),
            "pad_last": int(pad[i, -1]),
            "status_first5": status[i, :5, :].tolist(),
        })

    out = {
        "pad_unique_first30": np.unique(pad)[:30].tolist(),
        "pct_pad_matches_pkt_index_0_49": pct_eq_pktidx,
        "pad0_fraction_delta_eq_1": s0_inc1,
        "pad0_fraction_nondecreasing": s0_nondec,
        "padLast_fraction_delta_eq_1": sL_inc1,
        "padLast_fraction_nondecreasing": sL_nondec,
        "preview": preview,
        "status0_unique_count": int(np.unique(status[:, :, 0]).size),
        "status1_unique_count": int(np.unique(status[:, :, 1]).size),
        "status2_unique_count": int(np.unique(status[:, :, 2]).size),
    }
    return out


def decode_and_print_status_bytes(status: np.ndarray) -> dict:
    """
    Decode and print ADS1299 status bytes with CORRECT nibble reconstruction.
    
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  ADS1299 STATUS BYTE PACKING (3 bytes packed across nibbles)     ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  SB0[7:4] = LOFF_STATP[3:0]  (lower nibble of positive lead-off) ║
    ║  SB0[3:0] = LOFF_STATP[7:4]  (upper nibble of positive lead-off) ║
    ║  SB1[7:4] = LOFF_STATN[3:0]  (lower nibble of negative lead-off) ║
    ║  SB1[3:0] = LOFF_STATN[7:4]  (upper nibble of negative lead-off) ║
    ║  SB2[7:4] = Reserved/unused                                       ║
    ║  SB2[3:0] = GPIO[7:4]        (upper GPIO pins)                    ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    Parameters
    ----------
    status : np.ndarray
        Shape (n_frames, 50, 3) - status bytes for all packets
    
    Returns
    -------
    dict
        Decoded summary
    """
    print("\n" + "═" * 70)
    print(" ADS1299 STATUS BYTES ANALYSIS (CORRECTED DECODING) ".center(70, "═"))
    print("═" * 70)
    
    n_frames, n_packets, _ = status.shape
    total_samples = n_frames * n_packets
    
    # Extract raw status bytes
    SB0 = status[:, :, 0].flatten()  # Raw byte 0
    SB1 = status[:, :, 1].flatten()  # Raw byte 1
    SB2 = status[:, :, 2].flatten()  # Raw byte 2
    
    # ═══════════════════════════════════════════════════════════════════
    # CRITICAL: RECONSTRUCT ACTUAL REGISTER VALUES
    # ═══════════════════════════════════════════════════════════════════
    # The ADS1299 doesn't send full registers - it splits them across nibbles!
    LOFF_STATP = ((SB0 & 0x0F) << 4) | ((SB1 & 0xF0) >> 4)
    LOFF_STATN = ((SB1 & 0x0F) << 4) | ((SB2 & 0xF0) >> 4)
    GPIO_UPPER = (SB2 & 0x0F)
    
    print(f"\nTotal samples analyzed: {total_samples:,} ({n_frames} frames × {n_packets} packets)\n")
    
    # ═══ LOFF_STATP (Positive Lead-Off Detection) ═══
    print("─" * 70)
    print("LOFF_STATP: Lead-off Positive Electrode Detection (RECONSTRUCTED)")
    print("─" * 70)
    unique_statp = np.unique(LOFF_STATP)
    print(f"Unique values: {unique_statp.tolist()}")
    
    for val in unique_statp[:10]:  # Show top 10
        count = np.sum(LOFF_STATP == val)
        pct = 100.0 * count / total_samples
        binary = format(val, '08b')
        print(f"  0x{val:02X} ({binary}): {count:,} samples ({pct:.2f}%)")
        
        # Decode which channels have lead-off (bit N = channel N+1)
        if val == 0:
            print(f"    → All positive electrodes connected (or lead-off disabled)")
        else:
            leadoff_channels = [i+1 for i in range(8) if (val >> i) & 1]
            print(f"    → Lead-off detected on Ch{leadoff_channels}")
    
    # ═══ LOFF_STATN (Negative Lead-Off Detection) ═══
    print("\n" + "─" * 70)
    print("LOFF_STATN: Lead-off Negative Electrode Detection (RECONSTRUCTED)")
    print("─" * 70)
    unique_statn = np.unique(LOFF_STATN)
    print(f"Unique values: {unique_statn.tolist()}")
    
    for val in unique_statn[:10]:
        count = np.sum(LOFF_STATN == val)
        pct = 100.0 * count / total_samples
        binary = format(val, '08b')
        print(f"  0x{val:02X} ({binary}): {count:,} samples ({pct:.2f}%)")
        
        if val == 0:
            print(f"    → All negative electrodes connected (or lead-off disabled)")
        else:
            leadoff_channels = [i+1 for i in range(8) if (val >> i) & 1]
            print(f"    → Lead-off detected on Ch{leadoff_channels}")
    
    # ═══ GPIO Upper Nibble ═══
    print("\n" + "─" * 70)
    print("GPIO[7:4]: Upper GPIO Pin States")
    print("─" * 70)
    unique_gpio = np.unique(GPIO_UPPER)
    print(f"Unique values: {unique_gpio.tolist()}")
    
    for val in unique_gpio[:5]:
        count = np.sum(GPIO_UPPER == val)
        pct = 100.0 * count / total_samples
        binary = format(val, '04b')
        print(f"  0x{val:X} ({binary}): {count:,} samples ({pct:.2f}%)")
        print(f"    → GPIO[7:4] = {binary}")
    
    # ═══ RAW BYTES DEBUG INFO ═══
    print("\n" + "─" * 70)
    print("RAW STATUS BYTES (for debugging / verification)")
    print("─" * 70)
    print(f"SB0 unique: {np.unique(SB0)[:10].tolist()}")
    print(f"SB1 unique: {np.unique(SB1)[:10].tolist()}")
    print(f"SB2 unique: {np.unique(SB2)[:10].tolist()}")
    print("\nNOTE: These raw bytes are PACKED nibbles - see reconstruction above.")
    
    print("═" * 70 + "\n")
    
    # Return summary
    return {
        "loff_statp_values": unique_statp.tolist(),
        "loff_statn_values": unique_statn.tolist(),
        "gpio_values": unique_gpio.tolist(),
        "total_samples": total_samples,
        # Also include raw bytes for debugging
        "raw_sb0_values": np.unique(SB0).tolist(),
        "raw_sb1_values": np.unique(SB1).tolist(),
        "raw_sb2_values": np.unique(SB2).tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RAW CODE HEALTH CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def raw_code_summary(counts_all: np.ndarray, *, top_k: int = 10) -> dict:
    """
    Per-channel code distribution summary.
    
    Parameters
    ----------
    counts_all : np.ndarray
        Shape (8, n_samples) - raw ADC counts
    top_k : int
        Number of top values to include per channel
    
    Returns
    -------
    dict
        Summary with per-channel statistics
    """
    counts_all = np.asarray(counts_all)
    n_ch, n_samp = counts_all.shape
    out = {"n_ch": int(n_ch), "n_samples": int(n_samp), "channels": []}

    print(f"Raw counts summary: n_ch={n_ch}, n_samples={n_samp}\n")
    for ch in range(n_ch):
        x = counts_all[ch].astype(np.int32)
        p_zero = 100.0 * float(np.mean(x == 0))
        p_posfs = 100.0 * float(np.mean(x == FS_POS))
        p_negfs = 100.0 * float(np.mean(x == FS_NEG))

        vals, cnts = np.unique(x, return_counts=True)
        order = np.argsort(cnts)[::-1]
        vals = vals[order]
        cnts = cnts[order]
        n_unique = int(vals.size)

        top = []
        for v, c in zip(vals[:top_k], cnts[:top_k]):
            top.append({"value": int(v), "pct": 100.0 * float(c) / float(n_samp)})

        print(f"Ch{ch+1}: unique_codes={n_unique:,} | %zero={p_zero:.2f}% | %+FS={p_posfs:.2f}% | %-FS={p_negfs:.2f}%")
        for t in top:
            print(f"  {t['value']:>10d} : {t['pct']:6.2f}%")
        print()

        out["channels"].append({
            "ch": ch + 1,
            "unique_codes": n_unique,
            "pct_zero": p_zero,
            "pct_posfs": p_posfs,
            "pct_negfs": p_negfs,
            "top": top,
        })
    return out


def report_zero_entries(counts_all: np.ndarray) -> dict:
    """Count zero-value samples across all channels."""
    x = np.asarray(counts_all)
    total = int(x.size)
    zeros = int(np.sum(x == 0))
    pct = 100.0 * zeros / total if total else 0.0
    print(f"Zero entries: {zeros:,} / {total:,} ({pct:.6f}%)")
    return {"zeros": zeros, "total": total, "pct": pct}


def saturation_report(counts_all: np.ndarray) -> dict:
    """Check for ADC saturation (positive and negative full-scale)."""
    x = np.asarray(counts_all)
    total = int(x.size)
    pos = int(np.sum(x >= FS_POS))
    neg = int(np.sum(x <= FS_NEG))
    p_pos = 100.0 * pos / total if total else 0.0
    p_neg = 100.0 * neg / total if total else 0.0
    print(f"Saturation (>=+{FS_POS}): {pos:,} / {total:,} ({p_pos:.6f}%)")
    print(f"Saturation (<= {FS_NEG}): {neg:,} / {total:,} ({p_neg:.6f}%)")
    return {"pos": pos, "neg": neg, "total": total, "pct_pos": p_pos, "pct_neg": p_neg}
