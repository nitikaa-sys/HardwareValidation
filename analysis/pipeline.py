"""
Main Analysis Pipeline

Orchestrates the complete hardware validation analysis flow:
1. Parse filename metadata
2. Decode binary framestream
3. Raw data quality checks (before sorting)
4. Reorder frames chronologically
5. Post-sort quality checks
6. Generate plots
7. Export PDF report
"""

import copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .parse_filename import parse_filename_metadata, detect_test_type
from .decode_bin import (
    parse_ads1299_framestream_bin_bytes_strict_1416,
    duration_from_counts,
    N_CHANNELS,
)
from .raw_data_checks import (
    extract_packet_pad_and_status,
    quick_counter_report,
    decode_and_print_status_bytes,
    raw_code_summary,
    report_zero_entries,
    saturation_report,
)
from .timing_integrity import (
    unwrap_u32_to_i64,
    samples_sent_step_stats,
    dt1_metrics,
    estimate_dropped_samples_1sd,
    check_counter_near_rollover,
    reorder_by_t1,
    validate_timestamps,
    validate_t3_t2_physics,
    analyze_duplicate_frame_data,
)
from .preprocess import (
    preprocess_channel_uv,
    FS_HZ,
    BAND_MAX_HZ_DEFAULT,
)
from .plots import (
    plot_timing_signals,
    plot_timing_analysis,
    plot_frame_order_verification,
    print_dt1_tally_table,
    plot_channel_2x2,
    plot_all_channels_overlay,
)
from .report import (
    plot_analysis_summary_page,
    export_pdf,
)


def run_pipeline(
    source: Path | bytes,
    *,
    display_plots: bool = True,
    export_pdf_report: bool = True,
    out_dir: Path | None = None,
    source_name: str | None = None,
    # Configurable hardware parameters
    fs_hz: int | None = None,
    gain: int | None = None,
    mains_freq_hz: float = 50.0,
) -> dict:
    """
    Execute full hardware validation analysis pipeline.
    
    This is the main entry point that reproduces the notebook's
    analyze_bin_file() function behavior exactly.
    
    Supports both file path (offline) and raw bytes (online live capture).
    
    Parameters
    ----------
    source : Path or bytes
        Path to the .bin file OR raw frame bytes from live capture
    display_plots : bool
        If True, display plots interactively
    export_pdf_report : bool
        If True, export PDF report (only if source is Path)
    out_dir : Path, optional
        Output directory for PDF. Default: bin_path.parent / "reports"
    source_name : str, optional
        Display name for the source (used when source is bytes).
        Examples: "Internal Noise", "External Noise", "Live Capture"
    fs_hz : int, optional
        Sampling rate in Hz. Default: use FS_HZ from preprocess module (16000)
    gain : int, optional
        PGA gain setting. Default: use GAIN from preprocess module (24)
    mains_freq_hz : float
        Mains frequency for notch filter. Default: 50.0 (EU). Use 60.0 for US.
    
    Returns
    -------
    dict
        Complete analysis results including:
        - meta: parsed binary metadata
        - meta_raw: pre-sort metadata
        - filename_meta: parsed filename info
        - quality: channel quality checks
        - raw_checks: pre-sort validation results
        - was_sorted: whether frames were reordered
        - plots: list of generated figures
    """
    from datetime import datetime
    
    # Use defaults if not specified
    if fs_hz is None:
        fs_hz = FS_HZ
    
    # Determine if source is bytes or path
    is_bytes_source = isinstance(source, (bytes, bytearray))
    
    if is_bytes_source:
        # Live capture mode: source is raw bytes
        bin_bytes = bytes(source)
        bin_path = None
        
        # Build default metadata for live capture
        display_name = source_name if source_name else "Live Capture"
        filename_meta = {
            'firmware': 'FW2',
            'board': 'R2',
            'condition': source_name if source_name else 'LIVE',
            'date': datetime.now().strftime('%y%m%d'),
            'time': datetime.now().strftime('%H%M%S'),
            'n_frames_declared': 0,  # Unknown for live
            'sampling_rate_khz': 16,  # Default
            'display_name': display_name,
        }
        test_type = detect_test_type(filename_meta)
        
        print("=" * 80)
        print(" LIVE CAPTURE METADATA ".center(80, "="))
        print("=" * 80)
        print(f"Source:     {display_name}")
        print(f"Bytes:      {len(bin_bytes):,}")
        print(f"Timestamp:  {filename_meta['date']} {filename_meta['time']}")
        
        # Disable PDF export for bytes (no file path)
        if export_pdf_report:
            print("\n[NOTE] PDF export disabled for live capture (no file path)")
            export_pdf_report = False
    else:
        # File mode: source is path
        bin_path = Path(source)
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: PARSE METADATA (FIRST!)
        # ═══════════════════════════════════════════════════════════════════
        filename_meta = parse_filename_metadata(bin_path)
        test_type = detect_test_type(filename_meta)
        
        print("=" * 80)
        print(" FILE METADATA ".center(80, "="))
        print("=" * 80)
        print(f"File:       {bin_path.name}")
        print(f"Firmware:   {filename_meta['firmware']}")
        print(f"Board:      {filename_meta['board']}")
        print(f"Condition:  {filename_meta.get('condition', 'Unknown')}")
        print(f"Date:       {filename_meta['date']} {filename_meta.get('time', '')}")
        print(f"Declared:   {filename_meta['n_frames_declared']} frames @ {filename_meta['sampling_rate_khz']} kHz")
        
        # Load bytes from file
        bin_bytes = bin_path.read_bytes()
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: PARSE BINARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("PARSING BINARY")
    print("─" * 80)
    print(f"Loaded {len(bin_bytes):,} bytes")
    
    counts_all, meta = parse_ads1299_framestream_bin_bytes_strict_1416(bin_bytes)
    print(f"Parsed {meta['n_frames']} frames ({meta['n_samples']} samples)")
    print(f"Duration: {duration_from_counts(counts_all, fs_hz):.3f} s")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: RAW DATA QUALITY CHECKS (BEFORE ANY SORTING!)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(" RAW DATA QUALITY CHECKS (BEFORE SORTING) ".center(80, "="))
    print("=" * 80)
    print("(Analyzing data in RECEIVED ORDER - no sorting applied)\n")

    # CRITICAL: Compute dt1 directly from RAW meta
    hdr_raw = meta["hdr"]
    t1_raw_unwrapped = unwrap_u32_to_i64(hdr_raw["t1_first_drdy_us"])
    dt1_raw = np.diff(t1_raw_unwrapped)

    print(f"[Preprocessing] Computed {len(dt1_raw)} frame transitions from RAW data\n")

    # ═══ CHECK 1: DUPLICATE TIMESTAMPS (Δt1 = 0) ═══
    exact_zeros = int(np.sum(dt1_raw == 0))
    exact_zero_pct = 100.0 * exact_zeros / len(dt1_raw) if len(dt1_raw) > 0 else 0.0

    print(f"[Check 1] Duplicate Timestamps (Δt1 = 0):")
    print(f"    Count: {exact_zeros} ({exact_zero_pct:.2f}%)")

    if exact_zeros == 0:
        print(f"    Status: [OK] No duplicate timestamps")
    elif exact_zero_pct < 1.0:
        print(f"    Status: [WARNING] {exact_zeros} duplicate timestamps (<1%)")
    else:
        print(f"    Status: [ERROR] {exact_zeros} duplicate timestamps")

    # ═══ CHECK 2: DUPLICATE FRAME DATA ═══
    dup_data_check = analyze_duplicate_frame_data(counts_all, meta)
    print(f"\n[Check 2] Duplicate Frame Data Analysis:")
    if dup_data_check['has_duplicates']:
        print(f"    Duplicates with IDENTICAL data: {dup_data_check['identical_data_count']}")
        print(f"    Duplicates with DIFFERENT data: {dup_data_check['different_data_count']}")
        print(f"    Status: [{dup_data_check['severity']}] {dup_data_check['message']}")
        
        if dup_data_check['identical_data_count'] > 0:
            samples_lost = dup_data_check['identical_data_count'] * 50
            print(f"\n    ⚠️ CRITICAL: {dup_data_check['identical_data_count']} frames retransmitted!")
            print(f"    → DATA LOSS: ~{samples_lost} samples missing")
        
        if dup_data_check['different_data_count'] > 0:
            print(f"\n    ℹ️ Timestamp counter froze ({dup_data_check['different_data_count']} frames)")
            print(f"    → NO DATA LOSS: ADC still sampling correctly")
    else:
        print(f"    No duplicates to analyze")

    # ═══ CHECK 3: COUNTER ROLLOVER ═══
    rollover = check_counter_near_rollover(meta)
    print(f"\n[Check 3] Counter Rollover:")
    print(f"    Max t1: {rollover['max_t1']:,} µs")
    print(f"    Near rollover: {'⚠️ YES' if rollover['near_end'] else '✓ NO'}")
    print(f"    Safe to sort: {'✓ YES' if rollover['safe_to_sort'] else '❌ NO'}")

    # ═══ CHECK 4: FRAMES OUT OF ORDER (backward jumps) ═══
    backward_jumps = int(np.sum(dt1_raw < 0))
    print(f"\n[Check 4] Frames Out of Order (Δt1 < 0):")
    print(f"    Count: {backward_jumps} (backward time jumps)")

    # For compatibility with old code, create ts_check_raw dict
    ts_check_raw = {
        'negative_count': backward_jumps,
        'zero_count': exact_zeros,
    }

    # ═══ CHECK 5: t3 > t2 VIOLATIONS ═══
    physics_check = validate_t3_t2_physics(meta)
    print(f"\n[Check 5] t3 > t2 Violations (physical impossibility):")
    print(f"    Count: {physics_check['violation_count']} ({physics_check['violation_pct']:.2f}%)")
    if physics_check['violation_count'] > 0:
        print(f"    First violations at frames: {physics_check['violation_frames']}")
    print(f"    Status: [{physics_check['severity']}] {physics_check['message']}")

    # ═══ CHECK 6: Δt1 DISTRIBUTION SUMMARY (from RAW) ═══
    print(f"\n[Check 6] Δt1 Distribution (RAW data tally):")
    tally_raw = print_dt1_tally_table(meta, fs_hz=fs_hz)

    # ═══ SAVE RAW META FOR PLOTS (CRITICAL!) ═══
    meta_raw = copy.deepcopy(meta)  # Deep copy to preserve RAW state

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: DATA CLEANING (SORT FRAMES IF SAFE)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(" DATA CLEANING ".center(80, "="))
    print("=" * 80)

    was_sorted = False
    dropped_samples = {
        "mean_dt1_us": 0,
        "std_dt1_us": 0,
        "threshold_us": 0,
        "outlier_count": 0,
        "total_dropped_samples_est": 0,
        "dropped_per_frame": [],
    }
    
    if not rollover["safe_to_sort"]:
        print("❌ Counter rollover detected - CANNOT safely reorder")
        print("   Analysis will use file order (may be incorrect!)")
    else:
        print("✓ Sorting frames by t1 for display/analysis...")
        counts_all, meta = reorder_by_t1(counts_all, meta)
        was_sorted = True
        print("  ✓ Frames reordered chronologically")
        print("\n  [NOTE] All subsequent plots show SORTED data")

        # ═══ CHECK 7: DROPPED SAMPLES (AFTER SORTING!) ═══
        print("\n" + "─" * 80)
        print("POST-SORT QUALITY CHECK")
        print("─" * 80)
        
        dropped_samples = estimate_dropped_samples_1sd(meta, fs_hz=fs_hz)
        print(f"\n[Check 7] Dropped Samples Estimation (1 SD method):")
        print(f"    [NOTE] Calculated from SORTED timeline (negative Δt1 removed by sorting)")
        print(f"    Mean Δt1: {dropped_samples['mean_dt1_us']:.1f} µs")
        print(f"    Std Δt1: {dropped_samples['std_dt1_us']:.1f} µs")
        print(f"    Threshold (mean+1SD): {dropped_samples['threshold_us']:.1f} µs")
        print(f"    Outlier frames: {dropped_samples['outlier_count']}")
        print(f"    Total dropped samples estimate: {dropped_samples['total_dropped_samples_est']:,}")
        
        if dropped_samples['outlier_count'] > 0:
            print(f"\n    Outlier details:")
            for detail in dropped_samples.get('dropped_per_frame', [])[:5]:
                print(f"      Frame {detail['frame_index']}: "
                    f"Δt1={detail['dt1_us']:,} µs (excess: +{detail['excess_us']:,} µs) "
                    f"→ {detail['dropped_samples_est']} samples dropped")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: CHANNEL DATA QUALITY (SORTED)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(" CHANNEL DATA QUALITY (SORTED) ".center(80, "="))
    print("=" * 80)
    
    quality = {
        'raw_codes': raw_code_summary(counts_all, top_k=3),
        'zeros': report_zero_entries(counts_all),
        'saturation': saturation_report(counts_all),
    }

    # Add counter/pad analysis
    print("\n[Packet Counter Analysis]")
    pad, status = extract_packet_pad_and_status(bin_bytes)
    counter_report = quick_counter_report(pad, status)
    print(f"  Pad matches index [0-49]: {counter_report['pct_pad_matches_pkt_index_0_49']:.1f}%")
    quality['counter'] = counter_report
    status_decode = decode_and_print_status_bytes(status)
    quality['status_bytes'] = status_decode

    # Add samples_sent analysis
    print("\n[Samples Sent Analysis]")
    ss_stats = samples_sent_step_stats(meta)
    print(f"  Format: {ss_stats['format']}")
    print(f"  Steps = 50: {ss_stats['eq50']}/{ss_stats['n_transitions']}")
    print(f"  Est. missing frames: {ss_stats['missing_frames_est_from_forward_jumps']}")
    quality['samples_sent'] = ss_stats

    # ═══════════════════════════════════════════════════════════════════
    # STEP 6: GENERATE PLOTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 80)
    print("GENERATING PLOTS...")
    print("─" * 80)

    plots = []
    title = filename_meta['display_name']

    # Plot 0: SUMMARY PAGE (uses RAW check results)
    print("  0. Analysis summary page...")
    plots.append(plot_analysis_summary_page(
        bin_path=bin_path,
        filename_meta=filename_meta,
        meta=meta_raw,
        rollover=rollover,
        exact_zeros=exact_zeros,
        exact_zero_pct=exact_zero_pct,
        dup_data_check=dup_data_check,
        ts_check_raw=ts_check_raw,
        physics_check=physics_check,
        tally_raw=tally_raw,
        dropped_samples=dropped_samples,
        was_sorted=was_sorted,
    ))

    print("  1. Timing signals (t1, t2, t3)...")
    plots.append(plot_timing_signals(meta_raw, fs_hz=fs_hz))

    print("  2. Timing analysis (histogram + tally)...")
    plots.append(plot_timing_analysis(meta_raw, fs_hz=fs_hz))

    print("  3. Frame order verification...")
    plots.append(plot_frame_order_verification(meta))

    print("  4. All-channels overlay...")
    plots.append(plot_all_channels_overlay(
        counts_all,
        fs_hz=fs_hz,
        title_prefix=title,
        band_max_hz=BAND_MAX_HZ_DEFAULT,
        test_type=test_type,
    ))
    
    # Plots 5-12: Individual channels
    for ch in range(1, N_CHANNELS + 1):
        print(f"  {ch+4}. Channel {ch}...")
        uv_ch = preprocess_channel_uv(
            counts_all[ch-1],
            fs_hz=fs_hz,
            channel_idx=ch-1,
            apply_notch=True,
            test_type=test_type,
        )
        plots.append(plot_channel_2x2(
            ch=ch,
            uv_ch=uv_ch,
            counts_ch=counts_all[ch-1],
            fs_hz=fs_hz,
            title_prefix=title,
            band_max_hz=BAND_MAX_HZ_DEFAULT,
        ))
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 7: DISPLAY OR EXPORT
    # ═══════════════════════════════════════════════════════════════════
    if display_plots:
        for fig in plots:
            plt.show()
    
    if export_pdf_report:
        out_dir_path = Path(out_dir) if out_dir else bin_path.parent / "reports"
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        pdf_name = bin_path.stem + "_report.pdf"
        pdf_path = out_dir_path / pdf_name
        
        print(f"\n✓ Exporting PDF: {pdf_path.name}")
        export_pdf(plots, pdf_path, display_plots=display_plots)
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)
    
    return {
        'meta': meta,
        'meta_raw': meta_raw,
        'filename_meta': filename_meta,
        'quality': quality,
        'raw_checks': {
            'timestamp': ts_check_raw,
            'physics': physics_check,
            'duplicate_data': dup_data_check,
            'tally': tally_raw,
        },
        'was_sorted': was_sorted,
        'plots': plots,
    }
