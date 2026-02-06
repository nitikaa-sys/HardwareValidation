"""
Constants for ADS1299 acquisition system.

Separated into:
- PROTOCOL CONSTANTS: Hardware-defined, do not change
- DEFAULTS & POLICY: Configurable values
- CODE MAPPINGS: Filename encoding schemes
"""

# =============================================================================
# PROTOCOL CONSTANTS (hardware-defined, do not change)
# =============================================================================

# Frame format (matching C backend)
FRAME_HEADER_SIZE = 16       # 16-byte header
PACKET_SIZE = 28             # 27 data + 1 DMA padding
PACKET_DATA_SIZE = 27        # 3 status + 24 channel data
FRAME_TARGET_SIZE = 1416     # 16 + (50 × 28)
PACKETS_PER_FRAME = 50       # Fixed: 50 packets per frame
SAMPLES_PER_FRAME = 50       # Each packet = 1 sample
CHANNELS_PER_SAMPLE = 8      # ADS1299 has 8 channels

# Sample rate lookup from CONFIG1 ODR bits
SAMPLE_RATE_MAP = {
    0: 16000,
    1: 8000,
    2: 4000,
    3: 2000,
    4: 1000,
    5: 500,
    6: 250
}

# Gain lookup from CHnSET register
GAIN_MAP = {
    0: "1",
    1: "2",
    2: "4",
    3: "6",
    4: "8",
    5: "12",
    6: "24"
}


# =============================================================================
# DEFAULTS & POLICY (configurable)
# =============================================================================

# Event defaults
DEFAULT_FRAMES_PER_EVENT = 1665       # ~5 seconds at 16kHz
DEFAULT_SOUND_DURATION_US = 5000000   # 5 seconds in microseconds
DEFAULT_INTER_EVENT_DELAY = 0.5       # Delay between events (seconds)

# Timeouts
DEFAULT_TIMEOUT_NO_DATA = 10.0        # Stop if no data for this many seconds
DEFAULT_HTTP_TIMEOUT = 10.0           # HTTP request timeout
DEFAULT_WEBSOCKET_RECV_TIMEOUT = 0.1  # WebSocket recv poll timeout

# Concurrency
MAX_SAVE_WORKERS = 4                  # Max parallel background save threads

# Hardware defaults
DEFAULT_ESP32_IP = "node.local"
DEFAULT_SAMPLE_RATE = 16000

# LED colors
VALID_LED_COLORS = ['RED', 'BLUE', 'GREEN', 'YELLOW']


# =============================================================================
# CODE MAPPINGS (filename encoding schemes)
# =============================================================================

# Validation scenario codes (3-letter codes for filenames)
VALIDATION_SCENARIO_CODES = {
    "general_test": "GEN",
    "internal_noise": "INT",
    "external_noise": "EXT",
    "known_signal_injection": "KSI",
    "functional_tests": "FUN",
    "microphone_tests": "MIC"
}

# Experiment condition codes (3-letter codes for filenames)
EXPERIMENT_CONDITION_CODES = {
    "LEFT": "LEF",
    "RIGHT": "RIG",
    "MATH": "MTH",
    "REST": "RST"
}
