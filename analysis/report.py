"""
Report Generation

PDF summary page and export functionality.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_analysis_summary_page(
    bin_path: Path,
    filename_meta: dict,
    meta: dict,
    rollover: dict,
    exact_zeros: int,           
    exact_zero_pct: float,
    dup_data_check: dict,
    ts_check_raw: dict,
    physics_check: dict,
    tally_raw: dict,
    dropped_samples: dict,
    was_sorted: bool,
) -> plt.Figure:
    """
    Create a text-based summary page for PDF (page 1).
    Shows all critical checks and metadata.
    """
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.axis('off')
    
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("HARDWARE VALIDATION ANALYSIS REPORT".center(80))
    lines.append("=" * 80)
    lines.append("")
    
    # File metadata
    lines.append("FILE METADATA".center(80, "-"))
    lines.append(f"File:       {bin_path.name}")
    lines.append(f"Firmware:   {filename_meta['firmware']}")
    lines.append(f"Board:      {filename_meta['board']}")
    lines.append(f"Condition:  {filename_meta.get('condition', 'Unknown')}")
    lines.append(f"Date:       {filename_meta['date']} {filename_meta.get('time', '')}")
    lines.append(f"Declared:   {filename_meta['n_frames_declared']} frames @ {filename_meta['sampling_rate_khz']} kHz")
    lines.append("")
    
    # Binary info
    lines.append("BINARY DATA".center(80, "-"))
    lines.append(f"File size:  {bin_path.stat().st_size:,} bytes")
    lines.append(f"Parsed:     {meta['n_frames']:,} frames ({meta['n_samples']:,} samples)")
    lines.append("")
    
    # RAW DATA CHECKS (THE CRITICAL PART!)
    lines.append("=" * 80)
    lines.append("RAW DATA QUALITY CHECKS (BEFORE SORTING)".center(80))
    lines.append("=" * 80)
    lines.append("")

    lines.append("[1] Duplicate Timestamps (Δt1 = 0):")
    lines.append(f"    Count: {exact_zeros} ({exact_zero_pct:.2f}%)")
    if exact_zeros == 0:
        lines.append(f"    Status: [OK] No duplicate timestamps")
    elif exact_zero_pct < 1.0:
        lines.append(f"    Status: [WARNING] {exact_zeros} duplicate timestamps (<1%)")
    else:
        lines.append(f"    Status: [ERROR] {exact_zeros} duplicate timestamps")
    lines.append("")

    lines.append("[2] Duplicate Frame Data Analysis:")
    if dup_data_check['has_duplicates']:
        lines.append(f"    Identical data: {dup_data_check['identical_data_count']} frames")
        lines.append(f"    Different data: {dup_data_check['different_data_count']} frames")
        
        if dup_data_check['identical_data_count'] > 0:
            samples_lost = dup_data_check['identical_data_count'] * 50
            lines.append(f"    Status: [CRITICAL] DATA LOSS - ~{samples_lost} samples missing")
        elif dup_data_check['different_data_count'] > 0:
            lines.append(f"    Status: [ERROR] Frozen timestamp (no data loss)")
    else:
        lines.append("    No duplicates to analyze")
    lines.append("")

    lines.append("[3] Counter Rollover:")
    lines.append(f"    Max t1: {rollover['max_t1']:,} µs")
    lines.append(f"    Near rollover: {'⚠️ YES' if rollover['near_end'] else '✓ NO'}")
    lines.append(f"    Safe to sort: {'✓ YES' if rollover['safe_to_sort'] else '❌ NO'}")
    lines.append("")

    lines.append("[4] Frames Out of Order (Δt1 < 0):")
    lines.append(f"    Count: {ts_check_raw['negative_count']} (backward time jumps)")
    lines.append("")

    lines.append("[5] t3 > t2 Violations (physical impossibility):")
    lines.append(f"    Count: {physics_check['violation_count']} ({physics_check['violation_pct']:.2f}%)")
    if physics_check['violation_count'] > 0:
        lines.append(f"    First violations at frames: {physics_check['violation_frames']}")
    lines.append(f"    Status: [{physics_check['severity']}] {physics_check['message']}")
    lines.append("")

    lines.append("[6] Δt1 Distribution (RAW) - Top 10 Values:")
    if 'unique_values' in tally_raw and len(tally_raw['unique_values']) > 0:
        for i, (val, cnt, pct) in enumerate(zip(
            tally_raw['unique_values'][:10], 
            tally_raw['counts'][:10], 
            tally_raw['percentages'][:10]
        )):
            lines.append(f"    {int(val):>8} µs: {int(cnt):>6,} ({pct:5.2f}%)")
    lines.append("")

    # ═══ CHECK 7: Show it's calculated AFTER sorting ═══
    lines.append("")  # Extra space to separate from RAW checks
    lines.append("=" * 80)
    lines.append("POST-SORT QUALITY CHECK".center(80))
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("[7] Dropped Samples Estimation:")
    lines.append(f"    Method: 1 SD above mean Δt1 (calculated from SORTED timeline)")
    lines.append(f"    Threshold: {dropped_samples['threshold_us']:.1f} µs")
    lines.append(f"    Outlier frames: {dropped_samples['outlier_count']}")
    lines.append(f"    Estimated dropped samples: {dropped_samples['total_dropped_samples_est']:,}")
    lines.append("")
    
    # Data cleaning status
    lines.append("=" * 80)
    lines.append("DATA CLEANING".center(80))
    lines.append("=" * 80)
    lines.append("")
    if was_sorted:
        lines.append("✓ Frames SORTED by t1 for display")
        lines.append("  All subsequent plots show SORTED data")
    else:
        lines.append("❌ Frames NOT sorted (counter rollover detected)")
        lines.append("  All plots show data in RECEIVED order")
    lines.append("")
    
    # Report structure
    lines.append("=" * 80)
    lines.append("REPORT CONTENTS".center(80))
    lines.append("=" * 80)
    lines.append("")
    lines.append("Page 1:  Analysis Summary (this page)")
    lines.append("Page 2:  Timing Signals (t1, t2, t3)")
    lines.append("Page 3:  Frame Timing Analysis (Δt1 histogram + tally)")
    lines.append("Page 4:  Frame Order Verification")
    lines.append("Page 5:  All Channels Overlay")
    lines.append("Pages 6-13: Individual Channel Analysis (Ch1-Ch8)")
    
    # Display all text
    text = "\n".join(lines)
    ax.text(0.05, 0.95, text,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=9,
            family='monospace',
            transform=ax.transAxes)
    
    fig.suptitle("ADS1299 Hardware Analysis Report", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    return fig


def export_pdf(plots: list, output_path: Path, *, display_plots: bool = False) -> Path:
    """
    Export list of matplotlib figures to a multi-page PDF.
    
    Parameters
    ----------
    plots : list
        List of plt.Figure objects
    output_path : Path
        Output PDF file path
    display_plots : bool
        If True, keep figures open; if False, close after export
    
    Returns
    -------
    Path
        Path to the exported PDF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        for fig in plots:
            pdf.savefig(fig)
            if not display_plots:
                plt.close(fig)
    
    return output_path
