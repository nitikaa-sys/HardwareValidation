"""
Signal Preprocessing

ADC counts → µV conversion, DC removal, filtering, PSD/FFT computation.
"""

import numpy as np
from scipy import signal
from scipy.signal import decimate
from scipy.fft import rfft, rfftfreq

# ═══════════════════════════════════════════════════════════════════════════════
# ADC / DEVICE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

FS_HZ = 16000            # Hz
VREF_V = 4.5             # V (confirm for your board)
GAIN = 12                # ADS1299 PGA gain (confirm)
ADC_BITS = 24
CHANNEL_GAINS = [12, 12, 12, 12, 12, 12, 12, 12]  # Per-channel gains (Ch1-Ch8)

FS_COUNTS = (1 << (ADC_BITS - 1)) - 1            # 2^23 - 1 = 8388607

# Plot defaults
BAND_MAX_HZ_DEFAULT = FS_HZ / 2                  # 8000 Hz at 16 kHz
FFT_DBFS_FLOOR = -140.0


# ═══════════════════════════════════════════════════════════════════════════════
# COUNTS → MICROVOLTS CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def counts_to_uv(counts: np.ndarray, *, vref_v: float = VREF_V, gain: float | np.ndarray = None) -> np.ndarray:
    """
    ADS1299: Vin(V) ≈ counts * Vref / (gain * (2^23 - 1))
    
    Parameters
    ----------
    counts : np.ndarray
        Can be 1D (single channel) or 2D (multiple channels)
    vref_v : float
        Reference voltage in volts
    gain : float or np.ndarray
        Single gain (applied to all samples) or per-channel gain array
    
    Returns
    -------
    np.ndarray
        Signal in microvolts (µV)
    """
    if gain is None:
        gain = CHANNEL_GAINS if isinstance(counts, np.ndarray) and counts.ndim > 1 else GAIN
    
    x = np.asarray(counts, dtype=np.float64)
    
    # Handle per-channel gains for multi-channel input
    if isinstance(gain, (list, np.ndarray)):
        gain = np.array(gain)
        if x.ndim == 2:  # (channels, samples)
            gain = gain[:, np.newaxis]  # Broadcast to (channels, 1)
    
    return x * (vref_v * 1e6) / (gain * FS_COUNTS)


def preprocess_channel_uv(
    counts_ch: np.ndarray,
    *,
    fs_hz: int,
    channel_idx: int = 0,
    apply_notch: bool = True,
    notch_hz: float = 50.0,
    notch_q: float = 30.0,
    remove_dc: bool = True,
    test_type: str = "hardware",
) -> np.ndarray:
    """
    Full preprocessing for a single channel.
    
    Parameters
    ----------
    counts_ch : np.ndarray
        Raw ADC counts for one channel
    fs_hz : int
        Sampling rate in Hz
    channel_idx : int
        Channel index (0-7) for gain lookup
    apply_notch : bool
        Apply 50 Hz notch filter
    notch_hz : float
        Notch frequency
    notch_q : float
        Notch Q factor
    remove_dc : bool
        Remove DC offset
    test_type : str
        "hardware", "injection", or "functional"
        If "functional", applies 7-13 Hz bandpass for alpha analysis
    
    Returns
    -------
    np.ndarray
        Preprocessed signal in µV
    """
    # Use channel-specific gain
    gain = CHANNEL_GAINS[channel_idx] if channel_idx < len(CHANNEL_GAINS) else GAIN
    x = counts_to_uv(counts_ch, gain=gain)

    if test_type == "functional":
        nyquist = fs_hz / 2
        low = 7.0 / nyquist
        high = 13.0 / nyquist
        if 0 < low < high < 1.0 and x.size > 50:
            # Use SOS format for numerical stability with narrow bands
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            x = signal.sosfiltfilt(sos, x)
    
    if remove_dc:
        x = x - np.mean(x)
    
    if apply_notch and notch_hz > 0:
        b, a = signal.iirnotch(w0=notch_hz, Q=notch_q, fs=fs_hz)
        if x.size > 3 * max(len(a), len(b)):
            x = signal.filtfilt(b, a, x)
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# PSD & FFT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def welch_psd_uv(uv_ch: np.ndarray, *, fs_hz: int, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch PSD in µV²/Hz.
    
    Parameters
    ----------
    uv_ch : np.ndarray
        Signal in microvolts
    fs_hz : int
        Sampling rate in Hz
    nperseg : int, optional
        Segment length for Welch method
    
    Returns
    -------
    f : np.ndarray
        Frequency bins in Hz
    pxx : np.ndarray
        Power spectral density in µV²/Hz
    """
    x = np.asarray(uv_ch, dtype=np.float64)
    if nperseg is None:
        nperseg = min(4096, x.size) if x.size else 256
    nperseg = max(128, int(nperseg))
    f, pxx = signal.welch(x, fs=fs_hz, nperseg=nperseg, noverlap=nperseg//2, detrend="constant")
    return f, pxx  # µV²/Hz


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNSAMPLING FOR EEG
# ═══════════════════════════════════════════════════════════════════════════════

def downsample_for_eeg(
    signal_uv: np.ndarray,
    fs_original: int,
    fs_target: int = 250,
) -> tuple[np.ndarray, int]:
    """
    Properly downsample signal for EEG analysis with anti-alias filtering.
    
    Uses scipy.decimate which applies a Chebyshev anti-alias filter before
    decimation, maximizing SNR by averaging out high-frequency noise.
    
    SNR improvement: For 16kHz→250Hz (factor 64), gain is √64 = 8× (~18 dB)
    
    Parameters
    ----------
    signal_uv : np.ndarray
        Signal in microvolts at original sample rate
    fs_original : int
        Original sampling rate in Hz
    fs_target : int
        Target sampling rate in Hz (default 250 Hz, sufficient for EEG)
    
    Returns
    -------
    signal_ds : np.ndarray
        Downsampled signal
    fs_actual : int
        Actual sample rate after decimation
        
    Notes
    -----
    For EEG (0-100 Hz bandwidth), 250 Hz is more than adequate (Nyquist = 125 Hz).
    This gives ~18 dB SNR improvement vs. analyzing at 16 kHz.
    """
    if fs_original <= fs_target:
        return signal_uv, fs_original
    
    # Calculate decimation factor
    factor = int(fs_original / fs_target)
    
    if factor <= 1:
        return signal_uv, fs_original
    
    # scipy.decimate applies Chebyshev type I anti-alias filter then decimates
    # Use zero_phase=True for symmetric filtering (no phase distortion)
    # For large factors, do in stages to avoid numerical issues
    
    signal_ds = signal_uv.copy()
    fs_current = fs_original
    
    while factor > 1:
        # Max single-stage decimation factor (avoid numerical issues)
        stage_factor = min(factor, 10)
        signal_ds = decimate(signal_ds, stage_factor, zero_phase=True)
        fs_current = fs_current // stage_factor
        factor = factor // stage_factor
    
    return signal_ds, fs_current


def preprocess_channel_for_eeg(
    counts_ch: np.ndarray,
    *,
    fs_hz: int,
    channel_idx: int = 0,
    fs_target: int = 250,
    bandpass_hz: tuple = (1.0, 45.0),
    notch_hz: float = 50.0,
    notch_q: float = 30.0,
) -> tuple[np.ndarray, int]:
    """
    Full EEG preprocessing pipeline with proper downsampling for maximum SNR.
    
    Pipeline:
    1. Convert ADC counts to µV
    2. Remove DC offset
    3. Apply notch filter (at original rate for best attenuation)
    4. Downsample to target rate (with anti-alias filter)
    5. Apply bandpass filter (at downsampled rate)
    
    Parameters
    ----------
    counts_ch : np.ndarray
        Raw ADC counts for one channel
    fs_hz : int
        Original sampling rate in Hz
    channel_idx : int
        Channel index (0-7) for gain lookup
    fs_target : int
        Target sample rate after downsampling (default 250 Hz)
    bandpass_hz : tuple
        (low_hz, high_hz) for bandpass filter (default 1-45 Hz)
    notch_hz : float
        Power line notch frequency (50 Hz EU, 60 Hz US)
    notch_q : float
        Notch filter Q factor
    
    Returns
    -------
    signal_filtered : np.ndarray
        Preprocessed signal in µV at downsampled rate
    fs_out : int
        Output sample rate
    """
    # Step 1: Convert to µV
    gain = CHANNEL_GAINS[channel_idx] if channel_idx < len(CHANNEL_GAINS) else GAIN
    x = counts_to_uv(counts_ch, gain=gain)
    
    # Step 2: Remove DC
    x = x - np.mean(x)
    
    # Step 3: Notch filter at original rate (best attenuation)
    if notch_hz > 0 and fs_hz > 2 * notch_hz:
        b, a = signal.iirnotch(w0=notch_hz, Q=notch_q, fs=fs_hz)
        if x.size > 3 * max(len(a), len(b)):
            x = signal.filtfilt(b, a, x)
    
    # Step 4: Downsample (with anti-alias filter)
    x, fs_out = downsample_for_eeg(x, fs_hz, fs_target)
    
    # Step 5: Bandpass at downsampled rate
    low_hz, high_hz = bandpass_hz
    nyquist = fs_out / 2
    
    # Ensure valid normalized frequencies
    low_norm = max(0.001, low_hz / nyquist)
    high_norm = min(0.999, high_hz / nyquist)
    
    if 0 < low_norm < high_norm < 1.0 and x.size > 50:
        sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
        x = signal.sosfiltfilt(sos, x)
    
    return x, fs_out


def fft_dbfs_from_counts(counts_ch: np.ndarray, *, fs_hz: int, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """
    FFT magnitude as dBFS using raw counts as full-scale reference.
    dBFS = 20*log10(amp_counts / FS_COUNTS)
    
    Parameters
    ----------
    counts_ch : np.ndarray
        Raw ADC counts
    fs_hz : int
        Sampling rate in Hz
    window : str
        Window function ("hann" or "boxcar")
    
    Returns
    -------
    freqs : np.ndarray
        Frequency bins in Hz
    dbfs : np.ndarray
        FFT magnitude in dBFS
    """
    x = np.asarray(counts_ch, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    x = x - np.mean(x)

    if window == "hann":
        w = signal.windows.hann(n, sym=False)
    elif window == "boxcar":
        w = np.ones(n, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported window: {window}")

    xw = x * w
    spec = rfft(xw)
    freqs = rfftfreq(n, d=1.0 / fs_hz)

    denom = float(np.sum(w)) if np.sum(w) != 0 else float(n)
    amp = 2.0 * np.abs(spec) / denom
    if amp.size:
        amp[0] *= 0.5  # DC term

    dbfs = 20.0 * np.log10(amp / FS_COUNTS + 1e-20)
    dbfs = np.clip(dbfs, FFT_DBFS_FLOOR, 0.0)
    return freqs, dbfs
