"""
ESP32/ADS1299 hardware communication for acquisition system.

Provides HardwareClient class for all hardware interactions.
NO module-level global state - all state is encapsulated in client instance.

Classes:
- HardwareClient: Manages ESP32 connection and ADS1299 operations
"""

import json
import requests
from typing import Optional

from .constants import (
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_SAMPLE_RATE,
    SAMPLE_RATE_MAP,
    GAIN_MAP
)


class HardwareClient:
    """
    Client for ESP32/ADS1299 hardware communication.
    
    Encapsulates all hardware state and communication:
    - HTTP/WebSocket URLs
    - Timeouts
    - Current sample rate (detected from registers)
    
    Usage:
        client = HardwareClient("node.local")
        client.dump_and_verify_registers()
        client.load_and_apply_profile("profiles/all_shorted.json")
    """
    
    def __init__(self, ip: str, timeout: float = DEFAULT_HTTP_TIMEOUT):
        """
        Initialize hardware client.
        
        Args:
            ip: ESP32 IP address or hostname (e.g., "node.local", "192.168.1.100")
            timeout: HTTP request timeout in seconds
        """
        self.ip = ip
        self.http_url = f"http://{ip}"
        self.ws_url = f"ws://{ip}/ws"
        self.timeout = timeout
        self.sample_rate: Optional[int] = None  # Set after register dump
    
    def dump_and_verify_registers(self) -> dict:
        """
        Dump all ADS1299 registers and display for user verification.
        
        Calls GET /ads1299/diagnostics/registers-dump to read all 24 ADS1299 registers.
        Sets self.sample_rate based on CONFIG1 register.
        
        Returns:
            Dictionary of register values if successful, empty dict on failure
        """
        print(f"\n    Reading ADS1299 hardware registers...")
        
        try:
            response = requests.get(
                f"{self.http_url}/ads1299/diagnostics/registers-dump",
                timeout=self.timeout
            )
            if response.status_code == 200:
                registers = response.json()
                
                # Parse CONFIG1 for sample rate
                config1_hex = registers.get('CONFIG1', '0x00')
                config1_val = int(config1_hex, 16) if isinstance(config1_hex, str) else config1_hex
                odr_bits = config1_val & 0x07
                self.sample_rate = SAMPLE_RATE_MAP.get(odr_bits, DEFAULT_SAMPLE_RATE)
                sample_rate_str = str(self.sample_rate) if self.sample_rate > 0 else "unknown"
                
                # Parse CONFIG3 for internal reference
                config3_hex = registers.get('CONFIG3', '0x00')
                config3_val = int(config3_hex, 16) if isinstance(config3_hex, str) else config3_hex
                int_ref = "enabled" if (config3_val & 0x80) else "disabled"
                
                # Display register dump
                print(f"    ┌─────────────────────────────────────────────────┐")
                print(f"    │  ADS1299 Hardware Register Dump                 │")
                print(f"    ├─────────────────────────────────────────────────┤")
                print(f"    │  ID       = {registers.get('ID', 'N/A'):>6}  (Device ID)              │")
                print(f"    │  CONFIG1  = {registers.get('CONFIG1', 'N/A'):>6}  ({sample_rate_str} SPS)             │")
                print(f"    │  CONFIG2  = {registers.get('CONFIG2', 'N/A'):>6}  (Test signal config)     │")
                print(f"    │  CONFIG3  = {registers.get('CONFIG3', 'N/A'):>6}  (Int ref {int_ref})     │")
                print(f"    │  LOFF     = {registers.get('LOFF', 'N/A'):>6}  (Lead-off detect)        │")
                print(f"    ├─────────────────────────────────────────────────┤")
                
                # Channel settings
                for i in range(1, 9):
                    ch_key = f'CH{i}SET'
                    ch_hex = registers.get(ch_key, '0x00')
                    ch_val = int(ch_hex, 16) if isinstance(ch_hex, str) else ch_hex
                    gain_bits = (ch_val >> 4) & 0x07
                    gain = GAIN_MAP.get(gain_bits, "?")
                    pd = "OFF" if (ch_val & 0x80) else "ON"
                    print(f"    │  {ch_key}  = {ch_hex:>6}  (Gain={gain:>2}x, {pd:>3})         │")
                
                print(f"    ├─────────────────────────────────────────────────┤")
                print(f"    │  BIAS_SENSP = {registers.get('BIAS_SENSP', 'N/A'):>6}                        │")
                print(f"    │  BIAS_SENSN = {registers.get('BIAS_SENSN', 'N/A'):>6}                        │")
                print(f"    │  LOFF_SENSP = {registers.get('LOFF_SENSP', 'N/A'):>6}                        │")
                print(f"    │  LOFF_SENSN = {registers.get('LOFF_SENSN', 'N/A'):>6}                        │")
                print(f"    └─────────────────────────────────────────────────┘")
                
                # Verify device ID
                device_id = registers.get('ID', '0x00')
                if device_id in ['0x3E', '0x3e']:
                    print(f"    ✓ Device verified: ADS1299 (ID={device_id})")
                else:
                    print(f"    ⚠ Unexpected device ID: {device_id} (expected 0x3E for ADS1299)")
                
                return registers
            else:
                print(f"    ✗ Register dump failed: HTTP {response.status_code}")
                print(f"    Response: {response.text}")
                return {}
        except requests.exceptions.RequestException as e:
            print(f"    ✗ HTTP request failed: {e}")
            return {}
    
    def load_and_apply_profile(self, profile_path: str) -> bool:
        """
        Load an ADS1299 profile from a local JSON file and apply it to the ESP32.
        
        Args:
            profile_path: Path to the local JSON profile file
            
        Returns:
            True if profile was loaded and applied successfully, False otherwise
        """
        print(f"\n[PROFILE] Loading ADS1299 profile from: {profile_path}")
        
        # Step 1: Read local JSON file
        try:
            with open(profile_path, 'r') as f:
                file_profile = json.load(f)
            print(f"    ✓ Profile file loaded ({len(file_profile)} settings)")
            
            # Display key settings
            sample_rate = file_profile.get('CONFIG1_SAMPLE_RATE_SPS', 'N/A')
            int_ref = file_profile.get('CONFIG2_INT_REF_ENABLE', 'N/A')
            print(f"    Sample rate: {sample_rate} SPS")
            print(f"    Internal ref: {int_ref}")
            
            # Count enabled channels
            enabled_channels = sum(1 for i in range(1, 9) if file_profile.get(f'CH{i}_ENABLE', False))
            print(f"    Enabled channels: {enabled_channels}/8")
            
        except FileNotFoundError:
            print(f"    ✗ Profile file not found: {profile_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"    ✗ Invalid JSON in profile file: {e}")
            return False
        except Exception as e:
            print(f"    ✗ Failed to read profile file: {e}")
            return False
        
        # Step 2: Convert file format to API format
        print(f"    Converting profile to API format...")
        api_profile = self._convert_file_to_api_format(file_profile)
        print(f"    ✓ Converted ({len(api_profile['channels'])} channels configured)")
        
        # Step 3: Send profile to ESP32 via PUT /ads1299/profile
        try:
            print(f"    Uploading profile to ESP32...")
            response = requests.put(
                f"{self.http_url}/ads1299/profile",
                json=api_profile,
                timeout=self.timeout
            )
            if response.status_code in [200, 204]:
                print(f"    ✓ Profile uploaded successfully")
            else:
                print(f"    ✗ Profile upload failed: HTTP {response.status_code}")
                print(f"    Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"    ✗ HTTP request failed: {e}")
            return False
        
        # Step 4: Reload profile to apply to hardware registers
        try:
            print(f"    Applying profile to ADS1299 hardware...")
            response = requests.post(
                f"{self.http_url}/ads1299/profile/reload",
                timeout=self.timeout
            )
            if response.status_code in [200, 202]:
                print(f"    ✓ Profile applied to ADS1299 hardware registers")
                return True
            else:
                print(f"    ✗ Profile reload failed: HTTP {response.status_code}")
                print(f"    Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"    ✗ HTTP request failed: {e}")
            return False
    
    def prepare_routine(self, num_frames: int, led_color: str, sound_duration_us: int) -> bool:
        """
        Prepare routine control for streaming.
        
        Args:
            num_frames: Number of frames to acquire
            led_color: LED color to display (or "LED_OFF")
            sound_duration_us: Sound duration in microseconds
            
        Returns:
            True if preparation successful, False otherwise
        """
        try:
            payload = {
                "num_frames": num_frames,
                "led_color": led_color,
                "sound_duration_us": sound_duration_us
            }
            response = requests.post(
                f"{self.http_url}/api/v1/routines/control/prepare",
                json=payload,
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def execute_routine(self) -> bool:
        """
        Execute routine (deterministic start).
        
        Returns:
            True if execution started successfully, False otherwise
        """
        try:
            response = requests.post(
                f"{self.http_url}/api/v1/routines/control/execute",
                timeout=self.timeout
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def detect_sample_rate(self) -> Optional[int]:
        """
        Auto-detect sample rate from ESP32 register dump.
        
        Returns:
            Sample rate in Hz, or None if detection failed
        """
        try:
            response = requests.get(
                f"{self.http_url}/ads1299/diagnostics/registers-dump",
                timeout=self.timeout
            )
            if response.status_code == 200:
                registers = response.json()
                config1_hex = registers.get('CONFIG1', '0x00')
                config1_val = int(config1_hex, 16) if isinstance(config1_hex, str) else config1_hex
                odr_bits = config1_val & 0x07
                self.sample_rate = SAMPLE_RATE_MAP.get(odr_bits, DEFAULT_SAMPLE_RATE)
                return self.sample_rate
            return None
        except requests.exceptions.RequestException:
            return None
    
    def get_sample_rate(self) -> int:
        """
        Get current sample rate (from last register dump or default).
        
        Returns:
            Sample rate in Hz
        """
        return self.sample_rate if self.sample_rate else DEFAULT_SAMPLE_RATE
    
    @staticmethod
    def _convert_file_to_api_format(file_profile: dict) -> dict:
        """
        Convert file format profile to API format expected by PUT /ads1299/profile.
        
        File format uses field names like:
            CONFIG1_SAMPLE_RATE_SPS, CONFIG2_INT_REF_ENABLE, CH1_MODE, CH1_MUXP, etc.
        
        API format expects:
            sample_rate_sps, int_ref_enable, channels[] array with objects
            
        Args:
            file_profile: Profile dict from JSON file
            
        Returns:
            API-compatible profile dict
        """
        api_profile = {
            "sample_rate_sps": file_profile.get("CONFIG1_SAMPLE_RATE_SPS", 16000),
            "int_ref_enable": file_profile.get("CONFIG2_INT_REF_ENABLE", True),
            "bias_meas_enable": file_profile.get("CONFIG3_BIAS_MEAS_ENABLE", False),
            "loff_magnitude_ua": file_profile.get("LOFF_MAGNITUDE_uA", 6),
            "loff_freq_hz": file_profile.get("LOFF_FREQ_HZ", 32),
            "chop_enable": file_profile.get("CHOP_ENABLE", False),
            "channels": []
        }
        
        # Convert CH1-CH8 to channels array
        for i in range(1, 9):
            channel = {
                "mode": file_profile.get(f"CH{i}_MODE", "EEG"),
                "muxp": file_profile.get(f"CH{i}_MUXP", f"IN{i}P"),
                "muxn": file_profile.get(f"CH{i}_MUXN", "SRB1"),
                "gain": file_profile.get(f"CH{i}_GAIN", 24),
                "enable": file_profile.get(f"CH{i}_ENABLE", True)
            }
            api_profile["channels"].append(channel)
        
        return api_profile
