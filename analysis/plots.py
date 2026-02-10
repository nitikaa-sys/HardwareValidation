"""
Visualization Functions

All matplotlib figure generators for hardware validation analysis.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .decode_bin import N_CHANNELS
from .timing_integrity import unwrap_u32_to_i64, dt1_metrics
from .preprocess import (
    preprocess_channel_uv, welch_psd_uv, fft_dbfs_from_counts,
    preprocess_channel_for_eeg, downsample_for_eeg, counts_to_uv,
    FS_HZ, BAND_MAX_HZ_DEFAULT
)


# ═══════════════════════════════════════════════════════════════════════════════
# TIMING DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_timing_signals(meta: dict, *, fs_hz: int) -> plt.Figure:
    """
    Page 1: Core timing signals (t1, t2, t3).
    Simple 2-row layout.
    """
    hdr = meta["hdr"]
    t1 = unwrap_u32_to_i64(hdr["t1_first_drdy_us"])
    t2_us = hdr["t2_last_drdy_delta_4us"].astype(np.int64) * 4
    t3_us = hdr["t3_tx_ready_delta_4us"].astype(np.int64) * 4

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.35)
    
    ax1, ax_bottom = axes

    # Row 1: t1 unwrapped (full width)
    ax1.plot(t1, linewidth=1.0, color='steelblue')
    ax1.set_title(f"t1_first_drdy_us (unwrapped) | frames={meta['n_frames']:,}", 
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel("Frame index", fontsize=11)
    ax1.set_ylabel("t1 (µs)", fontsize=11)
    ax1.grid(alpha=0.3)

    # Row 2: Split into t2 (left) and t3 (right)
    ax_bottom.remove()
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 4)

    ax2.plot(t2_us, linewidth=0.8, alpha=0.7, color='#2196F3')
    ax2.set_title("t2: Last 2 DRDY Delta\n(ADC timing stability)", 
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel("Frame index", fontsize=10)
    ax2.set_ylabel("t2 (µs)", fontsize=10)
    ax2.grid(alpha=0.3)

    ax3.plot(t3_us, linewidth=0.8, alpha=0.7, color='#FF9800')
    ax3.set_title("t3: DRDY to TX_READY\n(Firmware response time)", 
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel("Frame index", fontsize=10)
    ax3.set_ylabel("t3 (µs)", fontsize=10)
    ax3.grid(alpha=0.3)

    fig.suptitle(f"TIMING SIGNALS | {meta['n_frames']} frames", 
                 fontsize=15, fontweight='bold', y=0.995)
    
    return fig


def plot_timing_analysis(meta: dict, *, fs_hz: int, jitter_tol_us: int = 200) -> plt.Figure:
    """
    Page 2: Frame timing analysis (histogram + tally table).
    Simple 2-row layout.
    """
    m = dt1_metrics(meta, fs_hz=fs_hz, jitter_tol_us=jitter_tol_us)
    dt1 = m["dt1"]
    expected = m["expected_dt1_us"]

    if dt1.size == 0:
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.text(0.5, 0.5, 'No timing data', ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig

    # Get unique values sorted by frequency
    unique_dt1, counts_dt1 = np.unique(dt1, return_counts=True)
    order = np.argsort(counts_dt1)[::-1]
    unique_dt1_sorted = unique_dt1[order]
    counts_dt1_sorted = counts_dt1[order]

    # Create figure with 2 rows
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.3, 1], hspace=0.4)

    # ═══ ROW 1: Δt1 Histogram ═══
    ax_hist = fig.add_subplot(gs[0])
    
    bins = np.arange(int(dt1.min()), int(dt1.max()) + 2, 1)
    ax_hist.hist(dt1, bins=bins, alpha=0.7, edgecolor='none', color='steelblue')
    ax_hist.axvline(expected, color='red', linestyle='--', linewidth=2.5, 
                    label=f"Expected={expected}µs", zorder=3)
    ax_hist.axvline(2 * expected, color='orange', linestyle='--', linewidth=2, 
                    label=f"2×={2*expected}µs", zorder=3)
    ax_hist.set_yscale('log')
    ax_hist.legend(fontsize=11, loc='upper right')
    ax_hist.set_title("Δt1 Histogram (Log Scale) - Frame-to-frame timing", 
                     fontweight='bold', fontsize=13)
    ax_hist.set_xlabel("Δt1 (µs)", fontsize=11)
    ax_hist.set_ylabel("Count (log scale)", fontsize=11)
    ax_hist.grid(alpha=0.3, which='both')

    # ═══ ROW 2: Tally Table ═══
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('tight')
    ax_table.axis('off')

    # Build table (top 15)
    n_show = min(15, len(unique_dt1_sorted))
    table_data = []
    
    for i in range(n_show):
        val = int(unique_dt1_sorted[i])
        cnt = int(counts_dt1_sorted[i])
        pct = 100.0 * cnt / dt1.size
        
        if val == 0:
            status = "⚠ DUPLICATE"
        elif abs(val - expected) < 50:
            status = "✓ Normal"
        elif abs(val - 2*expected) < 100:
            status = "⚠ 1 drop"
        elif val > 3 * expected:
            status = f"~{val//expected}×"
        else:
            status = "Outlier"
        
        table_data.append([f"{val:,}", f"{cnt:,}", f"{pct:.1f}%", status])

    # Create table
    table = ax_table.table(
        cellText=table_data,
        colLabels=['Δt1 (µs)', 'Count', '%', 'Status'],
        cellLoc='center',
        loc='center',
        colWidths=[0.22, 0.22, 0.18, 0.28]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Style table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white', fontsize=12)
        else:
            if 'DUPLICATE' in table_data[i-1][3]:
                cell.set_facecolor('#ffcccc')
            elif 'Normal' in table_data[i-1][3]:
                cell.set_facecolor('#e8f5e9')

    fig.suptitle(f"FRAME TIMING ANALYSIS | Δt1 Distribution & Tally", 
                 fontsize=15, fontweight='bold', y=0.995)
    
    return fig


def plot_frame_order_verification(meta: dict) -> plt.Figure:
    """
    DEDICATED plot answering: "Are frames received in chronological order?"
    Shows t1 differences to detect any backward time jumps.
    """
    # CRITICAL FIX: Always use stored unwrapped if available, with deep copy
    if "t1_unwrapped_us" in meta["hdr"]:
        t1_unwrapped = np.array(meta["hdr"]["t1_unwrapped_us"], copy=True)
    else:
        t1 = meta["hdr"]["t1_first_drdy_us"]
        t1_unwrapped = unwrap_u32_to_i64(t1)
    
    dt1 = np.diff(t1_unwrapped)
    
    # Check if all positive (chronological)
    all_forward = np.all(dt1 > 0)
    pct_forward = 100.0 * np.mean(dt1 > 0)
    
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Δt1 over frames (should be all positive)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(dt1, linewidth=1.0, marker='.', markersize=2)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero line')
    ax1.set_title(f"Frame Order Check [AFTER SORTING] | All Δt1 > 0? {all_forward} ({pct_forward:.1f}% forward)", 
              fontsize=14, fontweight='bold')

    ax1.set_xlabel("Frame Transition Index")
    ax1.set_ylabel("Δt1 (µs)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Close-up of any negative values (backward jumps)
    ax2 = fig.add_subplot(2, 1, 2)
    negative_idx = np.where(dt1 <= 0)[0]
    if len(negative_idx) > 0:
        ax2.scatter(negative_idx, dt1[negative_idx], color='red', s=50, zorder=3, label=f'{len(negative_idx)} backward jumps')
        ax2.set_title(f"⚠ Backward Time Jumps Detected ({len(negative_idx)} locations)", color='red')
        ax2.set_xlabel("Frame Index")
        ax2.set_ylabel("Δt1 (µs)")
        ax2.legend()
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, '✓ NO BACKWARD JUMPS\nAll frames in chronological order', 
                 ha='center', va='center', fontsize=16, color='green', fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def print_dt1_tally_table(meta: dict, *, fs_hz: int) -> dict:
    """
    Print professional Δt1 tally table with interpretations.
    Returns dict for programmatic access.
    """
    dtm = dt1_metrics(meta, fs_hz=fs_hz, jitter_tol_us=200)
    dt1 = dtm["dt1"]
    expected = dtm["expected_dt1_us"]
    
    if dt1.size == 0:
        print("No Δt1 data")
        return {}
    
    # Get counts
    unique_dt1, counts_dt1 = np.unique(dt1, return_counts=True)
    
    # Sort by count (most common first)
    order = np.argsort(counts_dt1)[::-1]
    unique_dt1 = unique_dt1[order]
    counts_dt1 = counts_dt1[order]
    
    # Print formatted table
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " Δt1 HISTOGRAM TALLY - Frame Timing Analysis ".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║ Expected Δt1: {expected} µs (for {fs_hz} Hz, 50 samples/frame)".ljust(68) + "║")
    print("╠" + "═" * 15 + "╦" + "═" * 12 + "╦" + "═" * 10 + "╦" + "═" * 28 + "╣")
    print(f"║ {'Δt1 (µs)':<15} ║ {'Count':>10} ║ {'%':>8} ║ {'Interpretation':<28} ║")
    print("╠" + "═" * 15 + "╬" + "═" * 12 + "╬" + "═" * 10 + "╬" + "═" * 28 + "╣")
    
    for val, cnt in zip(unique_dt1, counts_dt1):
        pct = 100.0 * cnt / dt1.size
        
        # Interpretation
        if abs(val - expected) < 10:
            interp = "✓ Normal timing"
        elif abs(val - 2*expected) < 20:
            interp = "⚠ Possible 1 frame drop"
        elif abs(val - 3*expected) < 30:
            interp = "⚠ Possible 2 frame drops"
        elif val > 5 * expected:
            multiples = val // expected
            interp = f"⚠ ~{multiples}× expected"
        elif val < 0:
            interp = "✗ BACKWARD TIME JUMP!"
        elif val < 0.5 * expected:
            interp = "✗ Too fast!"
        else:
            interp = ""
        
        print(f"║ {int(val):<15} ║ {int(cnt):>10,} ║ {pct:>7.2f}% ║ {interp:<28} ║")
    
    print("╚" + "═" * 15 + "╩" + "═" * 12 + "╩" + "═" * 10 + "╩" + "═" * 28 + "╝")
    
    # Summary statistics
    print(f"\n📊 SUMMARY:")
    print(f"  Total frame transitions: {dt1.size:,}")
    print(f"  Unique Δt1 values: {len(unique_dt1)}")
    print(f"  Min Δt1: {int(dt1.min()):,} µs")
    print(f"  Max Δt1: {int(dt1.max()):,} µs")
    print(f"  Median Δt1: {int(np.median(dt1)):,} µs")
    print(f"  Expected Δt1: {expected} µs")
    
    return {
        "unique_values": unique_dt1.tolist(),
        "counts": counts_dt1.tolist(),
        "percentages": (100.0 * counts_dt1 / dt1.size).tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_channel_2x2(
    *,
    ch: int,
    uv_ch: np.ndarray,
    counts_ch: np.ndarray,
    fs_hz: int,
    title_prefix: str,
    band_max_hz: float | None = None,
    max_plot_points: int = 8000,
) -> plt.Figure:
    """
    2×2 figure:
      [0,0] Time series (display-clipped)
      [0,1] Histogram (robust range)
      [1,0] PSD (Welch) full band by default
      [1,1] FFT (dBFS) full band by default
    """
    band_max_hz = float(band_max_hz if band_max_hz is not None else fs_hz / 2.0)
    uv_ch = np.asarray(uv_ch, dtype=float).reshape(-1)
    counts_ch = np.asarray(counts_ch, dtype=np.int32).reshape(-1)

    n = uv_ch.size
    t = np.arange(n) / float(fs_hz)

    stride = max(1, n // int(max_plot_points))
    t_plot = t[::stride]
    x_plot = uv_ch[::stride]

    lo = float(np.percentile(x_plot, 1.0))
    hi = float(np.percentile(x_plot, 99.0))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    x_disp = np.clip(x_plot, lo, hi)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_ts, ax_hist, ax_psd, ax_fft = axes.flatten()
    ksi_match = re.search(r"KSI-CH(\d+)@(\d+)Hz", title_prefix, re.IGNORECASE)
    if ksi_match and int(ksi_match.group(1)) == ch:
        fig.suptitle(
            f"{title_prefix} · Ch{ch} | fs={fs_hz} Hz | N={n:,} | ⚠️ KSI-CH{ksi_match.group(1)}@{ksi_match.group(2)}Hz",
            fontsize=13
        )
    else:
        fig.suptitle(f"{title_prefix} · Ch{ch} | fs={fs_hz} Hz | N={n:,}", fontsize=13)

    ax_ts.plot(t_plot, x_disp, linewidth=1.0)
    ax_ts.set_title(f"Time series (display-clipped; stride={stride})")
    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("µV")
    ax_ts.grid(alpha=0.3)

    h_lo = float(np.percentile(uv_ch, 0.5))
    h_hi = float(np.percentile(uv_ch, 99.5))
    if h_lo == h_hi:
        h_lo -= 1.0
        h_hi += 1.0
    ax_hist.hist(uv_ch, bins=120, range=(h_lo, h_hi), alpha=0.7)
    ax_hist.axvline(0.0, linestyle="--", linewidth=1.2, label="0 µV")
    ax_hist.axvline(float(np.mean(uv_ch)), linestyle="--", linewidth=1.2, label="mean")
    ax_hist.axvline(float(np.median(uv_ch)), linestyle="--", linewidth=1.2, label="median")
    ax_hist.set_title("Histogram (0.5–99.5 pct range)")
    ax_hist.set_xlabel("µV")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.25)
    ax_hist.legend(loc="upper right")

    f_psd, pxx = welch_psd_uv(uv_ch, fs_hz=fs_hz)
    ax_psd.semilogy(f_psd, pxx, linewidth=1.1)
    ax_psd.set_xlim(0, band_max_hz)
    ax_psd.set_title(f"PSD (Welch) 0–{band_max_hz:g} Hz")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (µV²/Hz)")
    ax_psd.grid(alpha=0.3)

    f_fft, dbfs = fft_dbfs_from_counts(counts_ch, fs_hz=fs_hz)
    ax_fft.plot(f_fft, dbfs, linewidth=1.1)
    ax_fft.set_xlim(0, band_max_hz)
    ax_fft.set_ylim(-120, 5)
    ax_fft.axhline(0.0, linestyle="--", linewidth=1.0)
    ax_fft.set_title(f"FFT magnitude (dBFS) 0–{band_max_hz:g} Hz")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Amplitude (dBFS)")
    ax_fft.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# EEG ELECTRODE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

# Hardware channel → Electrode name mapping
# counts_all[0] = Hardware Channel 1
# counts_all[1] = Hardware Channel 2  → Fp1
# counts_all[2] = Hardware Channel 3  → Fp2
# counts_all[3] = Hardware Channel 4  → O1
# counts_all[4] = Hardware Channel 5  → O2
ELECTRODE_MAP = {
    2: "Fp1",   # Frontal left  → Hardware Ch2 → array index 1
    3: "Fp2",   # Frontal right → Hardware Ch3 → array index 2
    4: "O1",    # Occipital left  → Hardware Ch4 → array index 3
    5: "O2",    # Occipital right → Hardware Ch5 → array index 4
}

# Channels to plot (hardware channel numbers, in display order)
FUNCTIONAL_CHANNELS = [2, 3, 4, 5]

# Row order for the montage (top to bottom)
MONTAGE_ORDER = ["Fp1", "Fp2", "O1", "O2"]


def plot_all_channels_overlay(counts_all, *, fs_hz, title_prefix, band_max_hz, test_type="hardware"):
    """All 8 channels in ONE plot for quick comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{title_prefix} | ALL 8 CHANNELS", fontsize=13)
    ax_time, ax_hist, ax_psd, ax_fft = axes.flatten()
    
    colors = plt.cm.tab10(range(N_CHANNELS))
    
    for ch in range(N_CHANNELS):
        uv_ch = preprocess_channel_uv(counts_all[ch], fs_hz=fs_hz, channel_idx=ch, apply_notch=True, test_type=test_type)
        
        # Time (downsampled)
        stride = max(1, uv_ch.size // 8000)
        t = np.arange(0, uv_ch.size, stride) / fs_hz
        ax_time.plot(t, uv_ch[::stride], linewidth=0.8, alpha=0.7, 
                     label=f'Ch{ch+1}', color=colors[ch])
        
        # Histogram
        ax_hist.hist(uv_ch, bins=50, alpha=0.3, label=f'Ch{ch+1}', color=colors[ch])
        
        # PSD
        f_psd, pxx = welch_psd_uv(uv_ch, fs_hz=fs_hz)
        ax_psd.semilogy(f_psd, pxx, linewidth=1.0, alpha=0.7, 
                        label=f'Ch{ch+1}', color=colors[ch])
        
        # FFT
        f_fft, dbfs = fft_dbfs_from_counts(counts_all[ch], fs_hz=fs_hz)
        ax_fft.plot(f_fft, dbfs, linewidth=1.0, alpha=0.7, 
                    label=f'Ch{ch+1}', color=colors[ch])
    
    # Format axes
    ax_time.set_xlabel("Time (s)"); ax_time.set_ylabel("µV")
    ax_time.legend(fontsize=8, ncol=2); ax_time.grid(alpha=0.3)
    ax_time.set_title("Time Series")
    
    ax_hist.set_xlabel("µV"); ax_hist.set_ylabel("Count")
    ax_hist.legend(fontsize=8, ncol=2); ax_hist.grid(alpha=0.3)
    ax_hist.set_title("Histogram")
    
    ax_psd.set_xlim(0, band_max_hz); ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (µV²/Hz)"); ax_psd.legend(fontsize=8, ncol=2)
    ax_psd.grid(alpha=0.3); ax_psd.set_title(f"PSD 0-{band_max_hz:.0f} Hz")
    
    ax_fft.set_xlim(0, band_max_hz); ax_fft.set_ylim(-120, 5)
    ax_fft.set_xlabel("Frequency (Hz)"); ax_fft.set_ylabel("dBFS")
    ax_fft.legend(fontsize=8, ncol=2); ax_fft.grid(alpha=0.3)
    ax_fft.set_title(f"FFT 0-{band_max_hz:.0f} Hz")
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTIONAL EEG: EYES OPEN / EYES CLOSED MONTAGE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eo_ec_montage(
    counts_eo: np.ndarray,
    counts_ec: np.ndarray,
    *,
    fs_hz: int,
    time_window_s: tuple = (0, 5),
    title: str = "Eyes Open vs Eyes Closed — EEG Montage",
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """
    Plot 4-channel EEG montage comparing Eyes Open vs Eyes Closed.
    
    Layout:
        Left column:  Eyes Open
        Right column: Eyes Closed
        Rows:         Fp1, Fp2, O1, O2 (top to bottom)
    
    Parameters
    ----------
    counts_eo : np.ndarray
        Raw ADC counts for Eyes Open, shape (8, n_samples)
    counts_ec : np.ndarray
        Raw ADC counts for Eyes Closed, shape (8, n_samples)
    fs_hz : int
        Sampling rate in Hz
    time_window_s : tuple
        (start, end) time window in seconds to display
    title : str
        Overall figure title
    figsize : tuple
        Figure size (width, height) in inches
    
    Returns
    -------
    plt.Figure
    """
    
    # Calculate sample indices for time window
    start_sample = int(time_window_s[0] * fs_hz)
    end_sample = int(time_window_s[1] * fs_hz)
    
    # Time axis in seconds
    n_samples = end_sample - start_sample
    t = np.arange(n_samples) / fs_hz + time_window_s[0]
    
    # Create figure with shared axes
    fig, axes = plt.subplots(
        nrows=4, 
        ncols=2, 
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    
    # Column titles
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    axes[0, 0].set_title("Eyes Open", fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].set_title("Eyes Closed", fontsize=14, fontweight='bold', pad=10)
    
    # Define channel order for rows (index in FUNCTIONAL_CHANNELS)
    channel_order = [2, 3, 4, 5]  # Hardware channel numbers in row order
    
    # Process and plot each channel
    for row, hw_ch in enumerate(channel_order):
        arr_idx = hw_ch - 1  # Convert hardware channel to array index
        electrode = ELECTRODE_MAP[hw_ch]
        
        # ─── Eyes Open (left column) ───────────────────────────────────────
        uv_eo = preprocess_channel_uv(
            counts_eo[arr_idx],
            fs_hz=fs_hz,
            channel_idx=arr_idx,
            test_type="functional",  # Applies 7-13 Hz bandpass + 50 Hz notch
        )
        uv_eo_window = uv_eo[start_sample:end_sample]
        
        axes[row, 0].plot(t, uv_eo_window, linewidth=0.8, color='#1f77b4')
        axes[row, 0].set_ylabel(electrode, fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')
        axes[row, 0].yaxis.set_label_coords(-0.02, 0.5)
        axes[row, 0].grid(alpha=0.3, linestyle='--')
        
        # ─── Eyes Closed (right column) ────────────────────────────────────
        uv_ec = preprocess_channel_uv(
            counts_ec[arr_idx],
            fs_hz=fs_hz,
            channel_idx=arr_idx,
            test_type="functional",  # Applies 7-13 Hz bandpass + 50 Hz notch
        )
        uv_ec_window = uv_ec[start_sample:end_sample]
        
        axes[row, 1].plot(t, uv_ec_window, linewidth=0.8, color='#1f77b4')
        axes[row, 1].grid(alpha=0.3, linestyle='--')
    
    # X-axis labels (bottom row only)
    axes[3, 0].set_xlabel("Time (s)", fontsize=12)
    axes[3, 1].set_xlabel("Time (s)", fontsize=12)
    
    # Add scale bar annotation (bottom right)
    y_min, y_max = axes[0, 0].get_ylim()
    scale_range = y_max - y_min
    axes[3, 1].annotate(
        f'Scale\n{scale_range:.0f} µV',
        xy=(1.02, 0.5),
        xycoords='axes fraction',
        fontsize=10,
        ha='left',
        va='center',
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.1, wspace=0.15)
    
    return fig


def plot_eo_ec_psd_comparison(
    counts_eo: np.ndarray,
    counts_ec: np.ndarray,
    *,
    fs_hz: int,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Compare PSD between Eyes Open and Eyes Closed for all 4 functional channels.
    Emphasizes alpha band (7-13 Hz) where EC should show enhanced power.
    
    Parameters
    ----------
    counts_eo, counts_ec : np.ndarray
        Raw ADC counts, shape (8, n_samples)
    fs_hz : int
        Sampling rate
    
    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, hw_ch in enumerate(FUNCTIONAL_CHANNELS):
        arr_idx = hw_ch - 1
        electrode = ELECTRODE_MAP[hw_ch]
        ax = axes[idx]
        
        # Eyes Open PSD (raw processing, no alpha filter)
        uv_eo = preprocess_channel_uv(
            counts_eo[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
            test_type="hardware",  # No bandpass - full spectrum
        )
        f_eo, psd_eo = welch_psd_uv(uv_eo, fs_hz=fs_hz)
        
        # Eyes Closed PSD
        uv_ec = preprocess_channel_uv(
            counts_ec[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
            test_type="hardware",
        )
        f_ec, psd_ec = welch_psd_uv(uv_ec, fs_hz=fs_hz)
        
        # Plot
        ax.semilogy(f_eo, psd_eo, linewidth=1.2, label='Eyes Open', color='#2196F3')
        ax.semilogy(f_ec, psd_ec, linewidth=1.2, label='Eyes Closed', color='#E91E63')
        
        # Highlight alpha band
        ax.axvspan(7, 13, alpha=0.2, color='green', label='Alpha (7-13 Hz)')
        
        ax.set_xlim(0, 50)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (µV²/Hz)")
        ax.set_title(f"{electrode} — PSD Comparison", fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    fig.suptitle("Eyes Open vs Closed — PSD Comparison (Alpha Enhancement)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLICATION-STYLE PLOTS (Clean, Paper-Ready)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eo_ec_publication_montage(
    counts_eo: np.ndarray,
    counts_ec: np.ndarray,
    *,
    fs_hz: int,
    duration_s: float = 5.0,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Publication-style EEG montage: Eyes Open | Eyes Closed side-by-side.
    
    Matches reference Figure (c):
    - 4 channels stacked vertically (Fp1, Fp2, O1, O2)
    - Eyes Open (0-5s) left, Eyes Closed (5-10s) right
    - Vertical separator line
    - Clean scale bar
    
    Parameters
    ----------
    counts_eo, counts_ec : np.ndarray
        Raw ADC counts, shape (8, n_samples)
    fs_hz : int
        Sampling rate in Hz
    duration_s : float
        Duration to show for each condition (default 5s)
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
    """
    n_samples = int(duration_s * fs_hz)
    total_samples = n_samples * 2  # EO + EC concatenated
    
    # Create figure with single axis per channel (stacked)
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, sharey=True)
    
    # Time axis: 0 to 10 seconds (EO: 0-5, EC: 5-10)
    t = np.arange(total_samples) / fs_hz
    
    channel_order = [2, 3, 4, 5]  # Hardware channels
    
    # Track y-limits for scale bar
    all_data = []
    
    for row, hw_ch in enumerate(channel_order):
        arr_idx = hw_ch - 1
        electrode = ELECTRODE_MAP[hw_ch]
        ax = axes[row]
        
        # Preprocess both conditions (alpha band 7-13 Hz)
        uv_eo = preprocess_channel_uv(
            counts_eo[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
            test_type="functional",
        )[:n_samples]
        
        uv_ec = preprocess_channel_uv(
            counts_ec[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
            test_type="functional",
        )[:n_samples]
        
        # Concatenate EO + EC
        uv_concat = np.concatenate([uv_eo, uv_ec])
        all_data.append(uv_concat)
        
        # Plot continuous trace
        ax.plot(t, uv_concat, linewidth=0.6, color='#1f77b4')
        
        # Channel label on left
        ax.set_ylabel(electrode, fontsize=11, fontweight='bold', rotation=0, 
                      ha='right', va='center', labelpad=20)
        
        # Remove spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(left=False, labelleft=False)  # Hide y-tick labels
        
        # Light grid
        ax.grid(alpha=0.2, linestyle='-', axis='x')
    
    # Vertical separator at 5s (between EO and EC)
    for ax in axes:
        ax.axvline(duration_s, color='cyan', linewidth=1.5, linestyle='-', alpha=0.8)
    
    # Set common y-limits based on all data
    all_flat = np.concatenate(all_data)
    y_margin = np.percentile(np.abs(all_flat), 99) * 1.2
    for ax in axes:
        ax.set_ylim(-y_margin, y_margin)
    
    # X-axis settings (bottom only)
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    axes[-1].set_xlim(0, duration_s * 2)
    axes[-1].set_xticks(np.arange(0, duration_s * 2 + 1, 1))
    
    # Column labels at top
    axes[0].text(duration_s / 2, y_margin * 1.1, "Eyes open", 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    axes[0].text(duration_s * 1.5, y_margin * 1.1, "Eyes closed", 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Scale bar (bottom right)
    scale_val = int(round(y_margin * 0.8, -1))  # Round to nearest 10
    if scale_val < 10:
        scale_val = int(round(y_margin * 0.8))
    
    # Draw scale bar line
    x_bar = duration_s * 2 - 0.3
    y_bar_bottom = -y_margin * 0.9
    y_bar_top = y_bar_bottom + scale_val
    axes[-1].plot([x_bar, x_bar], [y_bar_bottom, y_bar_top], 
                  color='black', linewidth=2, clip_on=False)
    axes[-1].text(x_bar + 0.15, (y_bar_bottom + y_bar_top) / 2, 
                  f"Scale\n{scale_val}", fontsize=9, va='center', ha='left')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, top=0.92)
    
    return fig


def plot_eo_ec_publication_psd(
    counts_eo: np.ndarray,
    counts_ec: np.ndarray,
    *,
    fs_hz: int,
    channel: int = 5,  # Default: O2 (occipital - best for alpha)
    freq_max: float = 30.0,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Publication-style PSD comparison: Linear x-axis, overlaid traces.
    
    Matches reference Figure (b):
    - Single plot with EO and EC overlaid
    - LINEAR frequency axis (0, 5, 10, 15, 20, 25, 30 Hz)
    - LINEAR y-axis (µV²/Hz)
    - Clean legend
    
    Parameters
    ----------
    counts_eo, counts_ec : np.ndarray
        Raw ADC counts, shape (8, n_samples)
    fs_hz : int
        Sampling rate
    channel : int
        Hardware channel to plot (default 5 = O2)
    freq_max : float
        Maximum frequency to display (default 30 Hz)
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
    """
    arr_idx = channel - 1
    
    # Preprocess (NO alpha bandpass - full spectrum for PSD)
    uv_eo = preprocess_channel_uv(
        counts_eo[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
        test_type="hardware",  # No bandpass filter
    )
    uv_ec = preprocess_channel_uv(
        counts_ec[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
        test_type="hardware",
    )
    
    # Compute PSD
    f_eo, psd_eo = welch_psd_uv(uv_eo, fs_hz=fs_hz, nperseg=min(4096, len(uv_eo)))
    f_ec, psd_ec = welch_psd_uv(uv_ec, fs_hz=fs_hz, nperseg=min(4096, len(uv_ec)))
    
    # Mask to frequency range
    mask_eo = f_eo <= freq_max
    mask_ec = f_ec <= freq_max
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with LINEAR y-axis (not log!)
    ax.plot(f_eo[mask_eo], psd_eo[mask_eo], linewidth=1.5, 
            label='Eyes open', color='#1f77b4')
    ax.plot(f_ec[mask_ec], psd_ec[mask_ec], linewidth=1.5, 
            label='Eyes closed', color='#d62728')
    
    # X-axis: linear scale with exact ticks
    ax.set_xlim(0, freq_max)
    ax.set_xticks(np.arange(0, freq_max + 1, 5))
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    
    # Y-axis: linear scale
    ax.set_ylim(0, None)  # Start from 0
    ax.set_ylabel("Power spectral density (µV²/Hz)", fontsize=11)
    
    # Legend in upper right
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    
    # Clean grid
    ax.grid(alpha=0.3, linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_eo_ec_publication(
    counts_eo: np.ndarray,
    counts_ec: np.ndarray,
    *,
    fs_hz: int,
    duration_s: float = 5.0,
    psd_channel: int = 5,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Combined publication figure: Montage + PSD in one figure.
    
    Layout matches reference:
    - Left: Electrode positions diagram (optional, skip for now)
    - Right top: PSD comparison (Panel b)
    - Right bottom: Time series montage (Panel c)
    
    Parameters
    ----------
    counts_eo, counts_ec : np.ndarray
        Raw ADC counts
    fs_hz : int
        Sampling rate
    duration_s : float
        Duration per condition
    psd_channel : int
        Channel for PSD plot
    figsize : tuple
        Figure size
    
    Returns
    -------
    plt.Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid: 2 rows - PSD on top, montage on bottom
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.35)
    
    # ═══ TOP: PSD Comparison ═══
    ax_psd = fig.add_subplot(gs[0])
    
    arr_idx = psd_channel - 1
    electrode = ELECTRODE_MAP.get(psd_channel, f"Ch{psd_channel}")
    
    # Preprocess
    uv_eo = preprocess_channel_uv(
        counts_eo[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
        test_type="hardware",
    )
    uv_ec = preprocess_channel_uv(
        counts_ec[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
        test_type="hardware",
    )
    
    # PSD
    f_eo, psd_eo = welch_psd_uv(uv_eo, fs_hz=fs_hz)
    f_ec, psd_ec = welch_psd_uv(uv_ec, fs_hz=fs_hz)
    
    freq_max = 30.0
    mask = f_eo <= freq_max
    
    ax_psd.plot(f_eo[mask], psd_eo[mask], linewidth=1.5, 
                label='Eyes open', color='#1f77b4')
    ax_psd.plot(f_ec[mask], psd_ec[mask], linewidth=1.5, 
                label='Eyes closed', color='#d62728')
    
    ax_psd.set_xlim(0, freq_max)
    ax_psd.set_xticks(np.arange(0, freq_max + 1, 5))
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_psd.set_ylim(0, None)
    ax_psd.set_ylabel("Power spectral density (µV²/Hz)", fontsize=11)
    ax_psd.legend(loc='upper right', fontsize=10)
    ax_psd.grid(alpha=0.3)
    ax_psd.spines['top'].set_visible(False)
    ax_psd.spines['right'].set_visible(False)
    ax_psd.set_title(f"(b) PSD Comparison — {electrode}", fontsize=12, fontweight='bold', loc='left')
    
    # ═══ BOTTOM: Time Series Montage ═══
    gs_montage = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1], hspace=0.05)
    
    n_samples = int(duration_s * fs_hz)
    total_samples = n_samples * 2
    t = np.arange(total_samples) / fs_hz
    
    channel_order = [2, 3, 4, 5]
    all_data = []
    montage_axes = []
    
    for row, hw_ch in enumerate(channel_order):
        ax = fig.add_subplot(gs_montage[row])
        montage_axes.append(ax)
        
        arr_idx = hw_ch - 1
        electrode_name = ELECTRODE_MAP[hw_ch]
        
        uv_eo_ch = preprocess_channel_uv(
            counts_eo[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
            test_type="functional",
        )[:n_samples]
        
        uv_ec_ch = preprocess_channel_uv(
            counts_ec[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
            test_type="functional",
        )[:n_samples]
        
        uv_concat = np.concatenate([uv_eo_ch, uv_ec_ch])
        all_data.append(uv_concat)
        
        ax.plot(t, uv_concat, linewidth=0.6, color='#1f77b4')
        ax.set_ylabel(electrode_name, fontsize=10, fontweight='bold', rotation=0, 
                      ha='right', va='center', labelpad=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
        ax.grid(alpha=0.2, axis='x')
        
        if row < 3:
            ax.tick_params(bottom=False, labelbottom=False)
    
    # Separator line
    for ax in montage_axes:
        ax.axvline(duration_s, color='cyan', linewidth=1.2, alpha=0.8)
    
    # Y-limits
    all_flat = np.concatenate(all_data)
    y_margin = np.percentile(np.abs(all_flat), 99) * 1.2
    for ax in montage_axes:
        ax.set_ylim(-y_margin, y_margin)
    
    # X-axis
    montage_axes[-1].set_xlabel("Time (s)", fontsize=11)
    montage_axes[-1].set_xlim(0, duration_s * 2)
    montage_axes[-1].set_xticks(np.arange(0, duration_s * 2 + 1, 1))
    
    # Labels
    montage_axes[0].text(duration_s / 2, y_margin * 1.05, "Eyes open", 
                         ha='center', va='bottom', fontsize=11, fontweight='bold')
    montage_axes[0].text(duration_s * 1.5, y_margin * 1.05, "Eyes closed", 
                         ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Scale bar
    scale_val = int(round(y_margin * 0.7, -1))
    if scale_val < 10:
        scale_val = int(round(y_margin * 0.7))
    montage_axes[-1].text(duration_s * 2 + 0.2, 0, f"Scale\n{scale_val}", 
                          fontsize=9, va='center', ha='left')
    
    montage_axes[0].set_title("(c) EEG Time Series", fontsize=12, fontweight='bold', loc='left')
    
    return fig


def plot_eo_ec_publication_complete(
    counts_eo: np.ndarray,
    counts_ec: np.ndarray,
    *,
    fs_hz: int,
    duration_s: float = 5.0,
    psd_channel: int = 5,
    montage_image_path: str = None,
    figsize: tuple = (14, 8),
    fs_target: int = 250,  # Target sample rate for SNR optimization
) -> plt.Figure:
    """
    Complete publication figure with 3 panels matching reference exactly.
    
    NOW WITH PROPER DOWNSAMPLING for maximum SNR!
    - 16kHz → 250Hz = √64 = 8× SNR improvement (~18 dB)
    
    Layout:
        (a) Left:         Electrode montage diagram (image)
        (b) Right-top:    PSD comparison (linear axes)
        (c) Right-bottom: Time series (4 channels stacked, EO|EC)
    
    Parameters
    ----------
    counts_eo, counts_ec : np.ndarray
        Raw ADC counts, shape (8, n_samples)
    fs_hz : int
        Original sampling rate in Hz
    duration_s : float
        Duration per condition (default 5s)
    psd_channel : int
        Hardware channel for PSD (default 5 = O2)
    montage_image_path : str or None
        Path to montage diagram PNG. If None, draws placeholder.
    figsize : tuple
        Figure size
    fs_target : int
        Target sample rate after downsampling (default 250 Hz)
        Use 250 Hz for EEG - gives SNR boost while keeping 0-100 Hz band
    
    Returns
    -------
    plt.Figure
    """
    import os
    from pathlib import Path
    
    fig = plt.figure(figsize=figsize)
    
    # Create main grid: 2 columns (montage left, PSD+timeseries right)
    gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.35, 0.65], wspace=0.15)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PANEL (a): Electrode Montage Diagram
    # ═══════════════════════════════════════════════════════════════════════════
    ax_montage = fig.add_subplot(gs_main[0])
    ax_montage.set_aspect('equal')
    ax_montage.axis('off')
    
    # Try to load montage image
    if montage_image_path and os.path.exists(montage_image_path):
        try:
            img = plt.imread(montage_image_path)
            ax_montage.imshow(img)
            ax_montage.set_title("(a)", fontsize=12, fontweight='bold', loc='left')
        except Exception as e:
            print(f"Warning: Could not load montage image: {e}")
            _draw_simple_montage(ax_montage)
    else:
        # Draw simple programmatic montage
        _draw_simple_montage(ax_montage)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RIGHT SIDE: PSD (top) + Time Series (bottom)
    # ═══════════════════════════════════════════════════════════════════════════
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_main[1], height_ratios=[1, 1.5], hspace=0.35
    )
    
    # ─── PANEL (b): PSD Comparison ─────────────────────────────────────────────
    # Now with PROPER DOWNSAMPLING for better SNR!
    ax_psd = fig.add_subplot(gs_right[0])
    
    arr_idx = psd_channel - 1
    electrode = ELECTRODE_MAP.get(psd_channel, f"Ch{psd_channel}")
    
    # Preprocess with downsampling for SNR (bandpass 1-45 Hz)
    uv_eo, fs_out_eo = preprocess_channel_for_eeg(
        counts_eo[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
        fs_target=fs_target,
        bandpass_hz=(1.0, 45.0),  # Full EEG range for PSD
    )
    uv_ec, fs_out_ec = preprocess_channel_for_eeg(
        counts_ec[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
        fs_target=fs_target,
        bandpass_hz=(1.0, 45.0),
    )
    
    # Compute PSD at downsampled rate
    f_eo, psd_eo = welch_psd_uv(uv_eo, fs_hz=fs_out_eo)
    f_ec, psd_ec = welch_psd_uv(uv_ec, fs_hz=fs_out_ec)
    
    freq_max = 30.0
    mask = f_eo <= freq_max
    
    # Plot with LINEAR y-axis (matching reference)
    ax_psd.plot(f_eo[mask], psd_eo[mask], linewidth=1.5, 
                label='Eyes open', color='#1f77b4')
    ax_psd.plot(f_ec[mask], psd_ec[mask], linewidth=1.5, 
                label='Eyes closed', color='#d62728')
    
    ax_psd.set_xlim(0, freq_max)
    ax_psd.set_xticks(np.arange(0, freq_max + 1, 5))
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_psd.set_ylim(0, None)
    ax_psd.set_ylabel("Power spectral density (V²/Hz)", fontsize=11)
    ax_psd.legend(loc='upper right', fontsize=10)
    ax_psd.grid(alpha=0.3)
    ax_psd.spines['top'].set_visible(False)
    ax_psd.spines['right'].set_visible(False)
    ax_psd.set_title("(b)", fontsize=12, fontweight='bold', loc='left')
    
    # ─── PANEL (c): Time Series Montage ────────────────────────────────────────
    # Now with PROPER DOWNSAMPLING for better SNR!
    gs_timeseries = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=gs_right[1], hspace=0.05
    )
    
    # Samples at downsampled rate
    n_samples_ds = int(duration_s * fs_target)
    total_samples_ds = n_samples_ds * 2
    t = np.arange(total_samples_ds) / fs_target  # Time axis at downsampled rate
    
    channel_order = [2, 3, 4, 5]  # Fp1, Fp2, O1, O2
    all_data = []
    ts_axes = []
    
    for row, hw_ch in enumerate(channel_order):
        ax = fig.add_subplot(gs_timeseries[row])
        ts_axes.append(ax)
        
        arr_idx = hw_ch - 1
        electrode_name = ELECTRODE_MAP[hw_ch]
        
        # Preprocess with downsampling for SNR + alpha bandpass
        uv_eo_ch, fs_out = preprocess_channel_for_eeg(
            counts_eo[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
            fs_target=fs_target,
            bandpass_hz=(7.0, 13.0),  # Alpha band for visualization
        )
        uv_eo_ch = uv_eo_ch[:n_samples_ds]
        
        uv_ec_ch, _ = preprocess_channel_for_eeg(
            counts_ec[arr_idx], fs_hz=fs_hz, channel_idx=arr_idx,
            fs_target=fs_target,
            bandpass_hz=(7.0, 13.0),
        )
        uv_ec_ch = uv_ec_ch[:n_samples_ds]
        
        # Concatenate EO + EC
        uv_concat = np.concatenate([uv_eo_ch, uv_ec_ch])
        all_data.append(uv_concat)
        
        # Plot continuous trace
        ax.plot(t, uv_concat, linewidth=0.6, color='#1f77b4')
        
        # Channel label (rotated 0°, left side)
        ax.set_ylabel(electrode_name, fontsize=11, fontweight='bold', rotation=0, 
                      ha='right', va='center', labelpad=15)
        
        # Clean axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False, labelleft=False)
        
        # Only bottom row has x-axis
        if row < 3:
            ax.tick_params(bottom=False, labelbottom=False)
            ax.spines['bottom'].set_visible(False)
    
    # Add vertical separator line at 5s (EO | EC boundary)
    for ax in ts_axes:
        ax.axvline(duration_s, color='#87CEEB', linewidth=1.5, alpha=0.9)
    
    # Set consistent y-limits across all channels
    all_flat = np.concatenate(all_data)
    y_margin = np.percentile(np.abs(all_flat), 99) * 1.2
    for ax in ts_axes:
        ax.set_ylim(-y_margin, y_margin)
    
    # X-axis settings (bottom row only)
    ts_axes[-1].set_xlabel("Time (s)", fontsize=11)
    ts_axes[-1].set_xlim(0, duration_s * 2)
    ts_axes[-1].set_xticks(np.arange(0, duration_s * 2 + 1, 1))
    ts_axes[-1].spines['bottom'].set_visible(True)
    
    # Column labels at top
    ts_axes[0].text(duration_s / 2, y_margin * 1.1, "Eyes open", 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ts_axes[0].text(duration_s * 1.5, y_margin * 1.1, "Eyes closed", 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Panel label
    ts_axes[0].set_title("(c)", fontsize=12, fontweight='bold', loc='left')
    
    # Scale bar (bottom right)
    scale_val = int(round(y_margin * 0.8, -1))
    if scale_val < 10:
        scale_val = max(1, int(round(y_margin * 0.8)))
    
    # Draw vertical scale bar
    x_bar = duration_s * 2 - 0.2
    y_bar_center = 0
    y_bar_half = scale_val / 2
    
    ts_axes[-1].plot([x_bar, x_bar], [y_bar_center - y_bar_half, y_bar_center + y_bar_half], 
                     color='black', linewidth=2, clip_on=False)
    ts_axes[-1].text(x_bar + 0.1, y_bar_center, f"Scale\n{scale_val}", 
                     fontsize=9, va='center', ha='left')
    
    plt.tight_layout()
    return fig


def _draw_simple_montage(ax):
    """
    Draw a simple programmatic electrode montage diagram.
    Shows the 10-20 head outline with Fp1, Fp2, O1, O2 highlighted.
    """
    # Head circle
    theta = np.linspace(0, 2*np.pi, 100)
    head_x = np.cos(theta)
    head_y = np.sin(theta)
    ax.plot(head_x, head_y, 'k-', linewidth=2)
    
    # Nose
    ax.plot([0, 0.1, 0], [1, 1.15, 1], 'k-', linewidth=2)
    
    # Ears
    ear_x = np.array([0.95, 1.05, 1.1, 1.05, 0.95])
    ear_y = np.array([0.1, 0.15, 0, -0.15, -0.1])
    ax.plot(ear_x, ear_y, 'k-', linewidth=2)
    ax.plot(-ear_x, ear_y, 'k-', linewidth=2)
    
    # 10-20 electrode positions (approximate)
    electrodes = {
        # Highlighted (red) - our 4 channels
        'Fp1': (-0.3, 0.75, 'red'),
        'Fp2': (0.3, 0.75, 'red'),
        'O1': (-0.3, -0.75, 'red'),
        'O2': (0.3, -0.75, 'red'),
        'GND': (0.0, 0.6, 'red'),
        'REF': (0.0, -0.4, 'red'),
        # Other positions (black)
        'F7': (-0.7, 0.5, 'black'),
        'F3': (-0.35, 0.45, 'black'),
        'Fz': (0.0, 0.45, 'black'),
        'F4': (0.35, 0.45, 'black'),
        'F8': (0.7, 0.5, 'black'),
        'T3': (-0.85, 0.0, 'black'),
        'C3': (-0.4, 0.0, 'black'),
        'Cz': (0.0, 0.0, 'black'),
        'C4': (0.4, 0.0, 'black'),
        'T4': (0.85, 0.0, 'black'),
        'T5': (-0.7, -0.5, 'black'),
        'P3': (-0.35, -0.45, 'black'),
        'Pz': (0.0, -0.45, 'black'),
        'P4': (0.35, -0.45, 'black'),
        'T6': (0.7, -0.5, 'black'),
    }
    
    for name, (x, y, color) in electrodes.items():
        if color == 'red':
            # Highlighted electrodes
            circle = plt.Circle((x, y), 0.08, color='red', fill=True, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y - 0.15, name, ha='center', va='top', fontsize=8, fontweight='bold')
        else:
            # Regular electrodes
            circle = plt.Circle((x, y), 0.06, color='black', fill=True, zorder=2)
            ax.add_patch(circle)
    
    # Dashed crosshairs
    ax.plot([-0.9, 0.9], [0, 0], 'k--', linewidth=1, alpha=0.5)
    ax.plot([0, 0], [-0.9, 0.9], 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title("(a)", fontsize=12, fontweight='bold', loc='left')
