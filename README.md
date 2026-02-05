# Hardware Validation System

A comprehensive system for acquiring and analyzing EEG data from hardware.

## Project Structure

```
HardwareAnalysis/
├── acquisition/                 # Live data acquisition (Web UI)
│   ├── web_ui.py               # FastAPI web interface
│   ├── run_experiment.py       # CLI experiment runner
│   ├── config.json             # Active configuration file
│   ├── profiles/               # Hardware register configurations
│   └── static/                 # Web UI HTML/CSS
│
├── analysis/                    # Offline analysis package
│   ├── __init__.py
│   ├── pipeline.py             # Main analysis orchestrator
│   ├── decode_bin.py           # Binary framestream parser
│   ├── preprocess.py           # Signal processing
│   ├── plots.py                # Visualization functions
│   ├── timing_integrity.py     # Frame timing analysis
│   ├── raw_data_checks.py      # Data quality checks
│   └── report.py               # PDF report generation
│
├── data/                        # Recorded .bin files (not tracked in git)
│
├── legacy/                      # Legacy/reference notebooks
│   └── offline_hardware_analysis.ipynb
│
└── hardware_validation.ipynb    # Main interactive analysis notebook
```

---

## Acquisition System (Data Collection)

The acquisition system provides live data collection from hardware via USB.

### Two Ways to Run Acquisition

| Method | Best For | Description |
|--------|----------|-------------|
| **Web UI** | Interactive use | Browser-based, click to configure and record |
| **CLI Script** | Automation, scripting | Command-line, batch experiments |

---

### Method 1: Web UI (Interactive)

Start the web interface:

```bash
cd acquisition
python web_ui.py
```

Then open `http://localhost:8000` in your browser.

**Features:**
- Real-time experiment configuration
- Profile selection from dropdown
- Start/stop recording with buttons
- Automatic .bin file generation

---

### Method 2: CLI Script (Automated)

For scripted/batch acquisition:

```bash
cd acquisition
python run_experiment.py --profile profiles/all_shorted.json --duration 10
```

**Options:**
```bash
python run_experiment.py --help
```

---

### Profiles

Pre-configured hardware register settings in `acquisition/profiles/`:

| Profile | Description |
|---------|-------------|
| `all_shorted.json` | Internal noise test (all inputs shorted) |
| `all_normal.json` | External noise test (floating inputs) |
| `test_signal.json` | Internal test signal generation |
| `ext_ch1_only.json` | Single channel external input |
| `eyes_open_closed.json` | Functional EEG configuration |

---

### Configuration (config.json)

The `acquisition/config.json` file controls all experiment settings.

**Structure:**

```json
{
  "esp32": {
    "ip": "node.local"
  },
  "experiment": {
    "num_events": 1,
    "random_seed": null,
    "profile_path": "profiles/eyes_open_closed.json",
    "output_folder": "260130",
    "inter_event_delay": 0.5
  },
  "routine": {
    "acquisition_duration_seconds": 10,
    "sound_duration_us": 0,
    "led_enabled": false,
    "frames_per_event": 200
  },
  "conditions": {
    "enforce_equal_condition_count": false,
    "mapping": {
      "RED": "VAL",
      "BLUE": "VAL",
      "GREEN": "VAL",
      "YELLOW": "VAL"
    }
  },
  "meta": {
    "mode": "validation",
    "firmware": "FW2",
    "board": "R2",
    "validation_scenario": "functional_tests"
  }
}
```

---

## Analysis System (Offline Processing)

The analysis system processes recorded .bin files and generates PDF reports.

### How to Run Analysis

Use the **interactive notebook** (recommended):

```bash
jupyter notebook hardware_validation.ipynb
```

**Steps:**
1. Run the **Setup** cell to import packages
2. Edit **File Paths** to point to your .bin files
3. Run any analysis cell

**Features:**
- See plots inline
- Easy parameter editing
- Markdown documentation for each test
- Eyes Open vs Eyes Closed comparison plots

---

### Python API

```python
from analysis.pipeline import run_pipeline
from pathlib import Path

# Run full analysis
results = run_pipeline(
    Path("data/260130_140604_FW2_R2_INT_1600_16k.bin"),
    display_plots=True,
    export_pdf_report=True,
)

# Access results
print(f"Frames: {results['meta']['n_frames']}")
print(f"Was sorted: {results['was_sorted']}")
```

---

## Test Categories

### Internal Noise (Shorted Inputs)
- **Purpose:** Measure ADC intrinsic noise floor
- **Setup:** All inputs shorted

### External Noise (Floating Inputs)
- **Purpose:** Measure environmental interference
- **Setup:** Inputs disconnected (floating)

### Known Signal Injection (KSI)
- **Purpose:** Verify channel integrity
- **Setup:** 40Hz signal injected into specific channel

### Functional EEG Tests
- **Purpose:** Real brain signal acquisition
- **Tests:** Eyes Open (FUNEO), Eyes Closed (FUNEC), Eye Blinks (FUNEB)

### Eyes Open vs Eyes Closed Montage
- Generates a graph of the power spectral density and time series for different electrode positions in the montage.

---

## Output

Each analysis generates:

1. **Console Output:** Real-time progress and quality metrics
2. **PDF Report:** Multi-page report with:
   - Summary page (metadata + quality checks)
   - Timing signals (t1, t2, t3)
   - Frame timing analysis
   - All-channels overlay
   - Individual channel analysis (8 pages)
3. **Results Dict:** Programmatic access to all data

Reports are saved to `{bin_file_directory}/reports/`

---

## Binary File Format

The framestream format:
- **Frame size:** 1416 bytes
- **Structure:** 16-byte header + 50 packets x 28 bytes
- **Samples per frame:** 50
- **Channels:** 8 x 24-bit ADC values

---

## ADC Configuration

### Parameters (in `analysis/preprocess.py`)

```python
FS_HZ = 16000        # Sampling rate (Hz)
VREF_V = 4.5         # Reference voltage (V)
GAIN = 12            # PGA gain
```

### Electrode Mapping (in `analysis/plots.py`)

```python
ELECTRODE_MAP = {
    2: "Fp1",   # Hardware Ch2 -> Frontal left
    3: "Fp2",   # Hardware Ch3 -> Frontal right
    4: "O1",    # Hardware Ch4 -> Occipital left
    5: "O2",    # Hardware Ch5 -> Occipital right
}
```

---

## Dependencies

```
numpy
scipy
matplotlib
fastapi (for acquisition/web_ui.py)
uvicorn
websockets
```

Install with:
```bash
pip install numpy scipy matplotlib fastapi uvicorn websockets
```

---

## File Naming Convention

```
YYMMDD_HHMMSS_FWX_RY_COND_NNNN_NNk.bin
```

| Field | Description |
|-------|-------------|
| YYMMDD | Date |
| HHMMSS | Time |
| FWX | Firmware version |
| RY | Board revision |
| COND | Test condition (INT, EXT, KSI_CHn, FUNEO, etc.) |
| NNNN | Frame count |
| NNk | Sampling rate in kHz |

Example: `260130_140604_FW2_R2_INT_1600_16k.bin`

---

## Troubleshooting

**Import errors:**
```bash
cd HardwareAnalysis
python -c "from analysis.pipeline import run_pipeline"
```

**File not found:**
- Check paths are correct (use raw strings on Windows: `r"path\to\file"`)
- Ensure .bin files exist in the specified location

**No plots shown:**
- Add `plt.show()` at the end of analysis
- Or use `display_plots=True` parameter

---

## Documentation

- `hardware_validation.ipynb` - Interactive analysis with explanations
- `legacy/offline_hardware_analysis.ipynb` - Legacy reference notebook
