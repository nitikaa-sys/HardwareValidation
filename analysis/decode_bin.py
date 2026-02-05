"""
Binary Frame Stream Decoder

Parse 1416-byte framestream binary files.
Extracts raw ADC counts and frame headers.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# STREAM FORMAT CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

N_CHANNELS = 8
STATUS_BYTES = 3
BYTES_PER_CH = 3
SAMPLE_BYTES = STATUS_BYTES + N_CHANNELS * BYTES_PER_CH          # 27

PACKETS_PER_FRAME = 50
PACKET_BYTES = 28                                               # 27 data + 1 pad
HEADER_BYTES = 16
FRAME_BYTES = HEADER_BYTES + PACKETS_PER_FRAME * PACKET_BYTES   # 1416


# ═══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _u16_le(b: np.ndarray) -> np.ndarray:
    """Read little-endian uint16 from 2-byte columns."""
    return (b[:, 0].astype(np.uint16) | (b[:, 1].astype(np.uint16) << 8))


def _u32_le(b: np.ndarray) -> np.ndarray:
    """Read little-endian uint32 from 4-byte columns."""
    return (
        b[:, 0].astype(np.uint32)
        | (b[:, 1].astype(np.uint32) << 8)
        | (b[:, 2].astype(np.uint32) << 16)
        | (b[:, 3].astype(np.uint32) << 24)
    )


def duration_from_counts(counts: np.ndarray, fs_hz: int) -> float:
    """Calculate recording duration from sample count."""
    n = int(counts.shape[1])
    return n / float(fs_hz) if fs_hz > 0 else float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BINARY PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_ads1299_framestream_bin_bytes_strict_1416(
    bin_bytes: bytes,
    *,
    require_t1_nonzero: bool = True,
    require_t1_nonzero_all_frames: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Strict 1416-byte framestream parser.
    
    Hard-fails on:
      - file not divisible by 1416
      - payload not exactly 1400 bytes/frame
      - packet_count != 50
      - t1 == 0 (frame 0 only, or all frames if configured)

    Parameters
    ----------
    bin_bytes : bytes
        Raw binary file contents
    require_t1_nonzero : bool
        If True, validate t1 timestamps are non-zero
    require_t1_nonzero_all_frames : bool
        If True, check all frames; if False, only check frame 0

    Returns
    -------
    counts_all : np.ndarray
        Shape (8, n_samples), int32 ADC counts
    meta : dict
        Metadata including:
        - frame_bytes, n_frames, n_samples
        - packets_per_frame, packet_bytes, sample_bytes
        - hdr: dict of header arrays (t1, t2, t3, packet_count, samples_sent, total_samples)
        - header_first, header_last: first/last header values
    
    Raises
    ------
    ValueError
        If binary format validation fails
    """
    data = np.frombuffer(bin_bytes, dtype=np.uint8)
    if data.size < FRAME_BYTES:
        raise ValueError(f"Bin too small: {data.size} bytes (< {FRAME_BYTES}).")

    if data.size % FRAME_BYTES != 0:
        raise ValueError(
            f"Bin length {data.size} not divisible by {FRAME_BYTES}. "
            "File may be truncated or different format."
        )

    n_frames = data.size // FRAME_BYTES
    frames = data.reshape(n_frames, FRAME_BYTES)

    header16 = frames[:, :HEADER_BYTES]
    packet_count = _u16_le(header16[:, 6:8])

    if not np.all(packet_count == PACKETS_PER_FRAME):
        bad_idx = np.where(packet_count != PACKETS_PER_FRAME)[0]
        i0 = int(bad_idx[0])
        raise ValueError(
            f"packet_count mismatch at frame {i0}: got {int(packet_count[i0])}, expected {PACKETS_PER_FRAME}."
        )

    hdr = {
        "t1_first_drdy_us": _u32_le(header16[:, 0:4]),
        "t2_last_drdy_delta_4us": header16[:, 4].astype(np.uint8),
        "t3_tx_ready_delta_4us": header16[:, 5].astype(np.uint8),
        "packet_count": packet_count,
        "samples_sent": _u32_le(header16[:, 8:12]),
        "total_samples": _u32_le(header16[:, 12:16]),
    }

    if require_t1_nonzero:
        t1 = hdr["t1_first_drdy_us"]
        if require_t1_nonzero_all_frames:
            zero_idx = np.where(t1 == 0)[0]
            if zero_idx.size:
                i0 = int(zero_idx[0])
                raise ValueError(f"t1_first_drdy_us == 0 at frame {i0}. Requirement violated.")
        else:
            if int(t1[0]) == 0:
                raise ValueError("t1_first_drdy_us == 0 at frame 0. Requirement violated.")

    payload = frames[:, HEADER_BYTES:]  # (n_frames, 1400)
    expected_payload_bytes = PACKETS_PER_FRAME * PACKET_BYTES
    if payload.shape[1] != expected_payload_bytes:
        raise ValueError(
            f"Payload size mismatch: got {payload.shape[1]} bytes/frame, expected {expected_payload_bytes}."
        )

    packets = payload.reshape(n_frames, PACKETS_PER_FRAME, PACKET_BYTES)

    # 27 data bytes, ignore 1 padding byte
    sample_bytes = packets[:, :, :SAMPLE_BYTES].reshape(-1, SAMPLE_BYTES)  # (n_samples, 27)

    # Drop 3 status bytes, parse 8 channels × 3 bytes (MSB first)
    ch_bytes = sample_bytes[:, STATUS_BYTES:].reshape(-1, N_CHANNELS, 3)

    x = (
        (ch_bytes[:, :, 0].astype(np.int32) << 16)
        | (ch_bytes[:, :, 1].astype(np.int32) << 8)
        |  ch_bytes[:, :, 2].astype(np.int32)
    )

    # Sign extend 24-bit two's complement
    x = np.where(x & 0x800000, x - 0x1000000, x).astype(np.int32)

    counts_all = x.T  # (8, n_samples)

    meta = {
        "frame_bytes": int(FRAME_BYTES),
        "n_frames": int(n_frames),
        "n_samples": int(counts_all.shape[1]),
        "packets_per_frame": int(PACKETS_PER_FRAME),
        "packet_bytes": int(PACKET_BYTES),
        "sample_bytes": int(SAMPLE_BYTES),
        "hdr": hdr,
        "header_first": {k: int(v[0]) for k, v in hdr.items()},
        "header_last": {k: int(v[-1]) for k, v in hdr.items()},
    }
    return counts_all, meta


# ═══════════════════════════════════════════════════════════════════════════════
# CLEAN ALIAS (use this in new code)
# ═══════════════════════════════════════════════════════════════════════════════

# Short alias for the main parsing function
parse_framestream_bin = parse_ads1299_framestream_bin_bytes_strict_1416
