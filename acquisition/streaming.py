"""
Frame streaming and experiment execution for acquisition system.

Classes:
- ExperimentEvent: Handles a single experiment event (streaming + save)

Functions:
- stream_event() - WebSocket frame reception
- background_save() - ThreadPoolExecutor save wrapper
- run_config_experiment() - Main experiment loop (config-driven)
- run_experiment() - Legacy experiment loop
"""

import asyncio
import json
import os
import struct
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Callable

import websockets

from .constants import (
    FRAME_HEADER_SIZE,
    FRAME_TARGET_SIZE,
    SAMPLES_PER_FRAME,
    DEFAULT_TIMEOUT_NO_DATA,
    DEFAULT_WEBSOCKET_RECV_TIMEOUT,
    DEFAULT_FRAMES_PER_EVENT,
    DEFAULT_SOUND_DURATION_US,
    DEFAULT_INTER_EVENT_DELAY,
    MAX_SAVE_WORKERS,
    VALID_LED_COLORS
)
from .hardware import HardwareClient
from .naming import (
    compute_output_folder,
    build_condition_code,
    build_sidecar_metadata,
    build_filename
)


async def capture_live_stream(
    client: HardwareClient,
    duration_seconds: float,
    sample_rate_sps: int = 16000,
    led_color: str = "LED_OFF",
    sound_duration_us: int = 0
) -> tuple:
    """
    Capture live EEG stream via WebSocket for online analysis.
    
    Uses the routines/control API (same as experiments).
    Returns raw frame bytes and info dict with timing headers.
    
    Args:
        client: HardwareClient instance (already connected)
        duration_seconds: Duration to capture in seconds
        sample_rate_sps: Sample rate in Hz (default 16000)
        led_color: LED color or "LED_OFF" (default "LED_OFF")
        sound_duration_us: Sound duration in microseconds (default 0)
        
    Returns:
        Tuple of (sample_bytes: bytes, raw_frames_bytes: bytes, info: dict)
        - sample_bytes: Extracted sample data (27 bytes/sample × N samples)
        - raw_frames_bytes: Raw 1416-byte frames concatenated
        - info: Dict with timing info, frame counts, header arrays
    """
    import struct
    from datetime import datetime
    
    # Calculate frames needed
    target_samples = int(round(sample_rate_sps * duration_seconds))
    target_frames = (target_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
    target_samples_exact = target_frames * SAMPLES_PER_FRAME
    
    print(f"[CAPTURE] Starting live stream capture...")
    print(f"    Duration: {duration_seconds}s")
    print(f"    Sample rate: {sample_rate_sps} SPS")
    print(f"    Target frames: {target_frames}")
    print(f"    Target samples: {target_samples_exact}")
    
    # Prepare routine
    print(f"[1/3] Preparing routine...")
    if not client.prepare_routine(target_frames, led_color, sound_duration_us):
        raise RuntimeError("Prepare routine failed")
    print(f"    Prepared")
    
    # Brief delay for hardware to settle
    await asyncio.sleep(0.2)
    
    # Execute routine
    print(f"[2/3] Executing routine...")
    if not client.execute_routine():
        raise RuntimeError("Execute routine failed")
    print(f"    Started streaming")
    
    # Brief delay for route activation
    await asyncio.sleep(0.3)
    
    # Receive frames via WebSocket
    print(f"[3/3] Connecting to WebSocket at {client.ws_url}...")
    
    # Storage
    raw_frames = []
    sample_buf = bytearray()
    
    # Header arrays for timing analysis
    t1_list, t2_list, t3_list = [], [], []
    pc_list, ss_list, ts_list = [], [], []
    
    # Counters
    recv_frames = 0
    rejected = {"bad_size": 0, "t1_zero": 0, "bad_packet_count": 0}
    timeouts = 0
    t_start = time.time()
    last_rx = time.time()
    
    try:
        async with websockets.connect(client.ws_url, ping_interval=20, ping_timeout=10) as ws:
            print(f"    WebSocket connected")
            
            while recv_frames < target_frames:
                # Check for timeout
                if time.time() - last_rx > DEFAULT_TIMEOUT_NO_DATA:
                    print(f"    Timeout: no data for {DEFAULT_TIMEOUT_NO_DATA}s")
                    break
                
                # Receive with timeout
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    timeouts += 1
                    continue
                
                if not isinstance(msg, (bytes, bytearray)):
                    continue
                
                last_rx = time.time()
                frame_size = len(msg)
                
                # Validate frame size
                if frame_size != FRAME_TARGET_SIZE:
                    rejected["bad_size"] += 1
                    continue
                
                frame = bytes(msg)
                
                # Parse 16-byte header
                t1 = struct.unpack_from('<I', frame, 0)[0]
                t2 = frame[4]
                t3 = frame[5]
                packet_count = struct.unpack_from('<H', frame, 6)[0]
                samples_sent = struct.unpack_from('<I', frame, 8)[0]
                total_samples = struct.unpack_from('<I', frame, 12)[0]
                
                # Validate t1 (should be non-zero)
                if t1 == 0:
                    rejected["t1_zero"] += 1
                    continue
                
                # Validate packet count
                if packet_count != SAMPLES_PER_FRAME:
                    rejected["bad_packet_count"] += 1
                    continue
                
                # Store header values
                t1_list.append(t1)
                t2_list.append(t2)
                t3_list.append(t3)
                pc_list.append(packet_count)
                ss_list.append(samples_sent)
                ts_list.append(total_samples)
                
                # Store raw frame
                raw_frames.append(frame)
                
                # Extract sample bytes (skip header, strip padding)
                payload = frame[FRAME_HEADER_SIZE:]  # 1400 bytes
                for i in range(SAMPLES_PER_FRAME):
                    pkt = payload[i * 28:(i + 1) * 28]
                    sample_buf.extend(pkt[:27])  # 27 bytes per sample (drop 1 padding)
                
                recv_frames += 1
                
                # Progress output every 100 frames
                if recv_frames % 100 == 0:
                    elapsed = time.time() - t_start
                    fps = recv_frames / elapsed if elapsed > 0 else 0
                    samples_so_far = len(sample_buf) // 27
                    print(f"    Frame {recv_frames}/{target_frames} | {fps:.0f} fps | {samples_so_far} samples")
    
    except websockets.exceptions.ConnectionClosed as e:
        print(f"    WebSocket closed: {e}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Build info dict
    elapsed = time.time() - t_start
    received_samples = len(sample_buf) // 27
    
    import numpy as np
    
    info = {
        "fs_hz": sample_rate_sps,
        "duration_s_requested": duration_seconds,
        "duration_s_measured": elapsed,
        "target_frames": target_frames,
        "target_samples_exact": target_samples_exact,
        "received_frames": recv_frames,
        "received_samples": received_samples,
        "received_bytes": len(sample_buf),
        "timeouts": timeouts,
        "rejected_frames": dict(rejected),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "samples_per_frame": SAMPLES_PER_FRAME,
        "frame_bytes": FRAME_TARGET_SIZE,
        "hdr": {
            "t1_first_drdy_us": np.asarray(t1_list, dtype=np.uint32),
            "t2_last_drdy_delta_4us": np.asarray(t2_list, dtype=np.uint8),
            "t3_tx_ready_delta_4us": np.asarray(t3_list, dtype=np.uint8),
            "packet_count": np.asarray(pc_list, dtype=np.uint16),
            "samples_sent": np.asarray(ss_list, dtype=np.uint32),
            "total_samples": np.asarray(ts_list, dtype=np.uint32),
        }
    }
    
    print(f"[CAPTURE] Complete: {recv_frames} frames, {received_samples} samples in {elapsed:.2f}s")
    if rejected["bad_size"] or rejected["t1_zero"] or rejected["bad_packet_count"]:
        print(f"    Rejected: {rejected}")
    
    # Return sample bytes, raw frames bytes, and info
    raw_frames_bytes = b''.join(raw_frames)
    return bytes(sample_buf), raw_frames_bytes, info


class ExperimentEvent:
    """
    Handles a single experiment event (streaming + save).
    
    Manages the lifecycle of a single acquisition:
    - Prepare routine on hardware
    - Execute streaming
    - Receive frames
    - Save binary + sidecar
    """
    
    def __init__(
        self,
        event_num: int,
        led_color: str,
        condition_label: str,
        config: dict,
        client: HardwareClient
    ):
        """
        Initialize experiment event.
        
        Args:
            event_num: Event number (1-indexed)
            led_color: LED color to display
            condition_label: Condition label from schedule
            config: Configuration dictionary
            client: HardwareClient instance for hardware communication
        """
        self.event_num = event_num
        self.led_color = led_color
        self.condition_label = condition_label
        self.config = config
        self.client = client
        
        # State tracking
        self.frames_received = 0
        self.samples_received = 0
        self.bytes_received = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.raw_frames: List[bytes] = []
        self.last_frame_time: Optional[float] = None
        self.success = False
        self.error_message: Optional[str] = None
        self.filename: Optional[str] = None
    
    def parse_frame_header(self, data: bytes) -> Optional[dict]:
        """Parse 16-byte frame header."""
        if len(data) < FRAME_HEADER_SIZE:
            return None
        
        t1_first_drdy_us = struct.unpack('<I', data[0:4])[0]
        t2_last_drdy_delta = data[4]
        t3_tx_ready_delta = data[5]
        packet_count = struct.unpack('<H', data[6:8])[0]
        samples_sent = struct.unpack('<I', data[8:12])[0]
        total_samples = struct.unpack('<I', data[12:16])[0]
        
        return {
            'packet_count': packet_count,
            'samples_sent': samples_sent,
            'total_samples': total_samples
        }
    
    async def prepare_routine(self) -> bool:
        """Prepare routine control for this event."""
        frames_per_event = self.config['routine']['frames_per_event']
        sound_duration_us = self.config['routine']['sound_duration_us']
        led_enabled = self.config['routine']['led_enabled']
        
        # LED control: send event color OR "LED_OFF"
        led_color = self.led_color if led_enabled else "LED_OFF"
        
        if self.client.prepare_routine(frames_per_event, led_color, sound_duration_us):
            return True
        else:
            self.error_message = "Prepare failed"
            return False
    
    async def execute_routine(self) -> bool:
        """Execute routine (deterministic start)."""
        if self.client.execute_routine():
            self.start_time = time.time()
            return True
        else:
            self.error_message = "Execute failed"
            return False
    
    async def receive_frames(
        self,
        websocket,
        experiment_start: float,
        status_cb: Optional[Callable] = None,
        stop_flag: Optional[threading.Event] = None
    ):
        """Receive frames via WebSocket with debug output."""
        self.last_frame_time = time.time()
        first_frame_time = None
        last_debug_frame = 0
        
        expected_frames = self.config['routine']['frames_per_event']
        
        try:
            while True:
                # Check for timeout
                if time.time() - self.last_frame_time > DEFAULT_TIMEOUT_NO_DATA:
                    self.error_message = f"Timeout: no data for {DEFAULT_TIMEOUT_NO_DATA}s"
                    break
                
                # Receive with timeout
                try:
                    data = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=DEFAULT_WEBSOCKET_RECV_TIMEOUT
                    )
                    self.last_frame_time = time.time()
                    
                    if first_frame_time is None:
                        first_frame_time = self.last_frame_time
                        
                except asyncio.TimeoutError:
                    continue
                
                # Process binary frame
                if isinstance(data, bytes):
                    frame_size = len(data)
                    
                    if frame_size == FRAME_TARGET_SIZE:
                        header = self.parse_frame_header(data)
                        if header:
                            self.frames_received += 1
                            self.samples_received = header['samples_sent']
                            self.bytes_received += frame_size
                            self.raw_frames.append(data)
                            
                            # Debug output every 200 frames
                            if self.frames_received - last_debug_frame >= 200:
                                elapsed = time.time() - first_frame_time if first_frame_time else 0
                                fps = self.frames_received / elapsed if elapsed > 0 else 0
                                t_offset = time.time() - experiment_start
                                print(f"    [E{self.event_num}] T+{t_offset:.1f}s | Frame {self.frames_received}/{expected_frames} | "
                                      f"{fps:.0f} fps | {self.bytes_received/1024:.0f} KB | pkts={header['packet_count']}")
                                
                                # Send frame progress update with exception handling
                                if status_cb:
                                    try:
                                        status_cb({
                                            "type": "frame_progress",
                                            "event": self.event_num,
                                            "frames_received": self.frames_received,
                                            "expected_frames": expected_frames
                                        })
                                    except Exception:
                                        pass  # Never break acquisition for UI reporting
                                
                                last_debug_frame = self.frames_received
                            
                            # Check if complete
                            if self.frames_received >= expected_frames:
                                self.success = True
                                break
                        
        except websockets.exceptions.ConnectionClosed:
            self.error_message = "WebSocket connection closed"
        except Exception as e:
            self.error_message = f"Receive error: {e}"
        
        self.end_time = time.time()
        
        # Final debug output
        duration = self.end_time - self.start_time if self.start_time else 0
        t_offset = time.time() - experiment_start
        if self.success:
            print(f"    [E{self.event_num}] T+{t_offset:.1f}s | COMPLETE | {self.frames_received} frames in {duration:.2f}s")
        else:
            print(f"    [E{self.event_num}] T+{t_offset:.1f}s | FAILED | {self.frames_received}/{expected_frames} frames | {self.error_message}")
    
    def save_binary(self, output_folder: str) -> bool:
        """
        Save raw binary + sidecar JSON with strict parser-compatible naming.
        
        Returns:
            True if save successful, False otherwise
        """
        if len(self.raw_frames) == 0:
            self.error_message = "No frames to save"
            return False
        
        meta = self.config.get('meta')
        frames = self.config['routine']['frames_per_event']
        sample_rate = self.client.get_sample_rate()
        
        # Build filename
        base_name = build_filename(
            self.led_color,
            self.condition_label,
            frames,
            sample_rate,
            meta
        )
        self.filename = f"{base_name}.bin"
        
        bin_path = os.path.join(output_folder, self.filename)
        json_path = os.path.join(output_folder, f"{base_name}.json")
        
        # Create folder if needed (handles nested paths)
        os.makedirs(output_folder, exist_ok=True)
        
        # Write binary file
        try:
            with open(bin_path, 'wb') as f:
                for frame in self.raw_frames:
                    f.write(frame)
        except Exception as e:
            self.error_message = f"Binary save failed: {e}"
            return False
        
        # Write sidecar JSON (warn if fails, but don't abort)
        try:
            sidecar = build_sidecar_metadata(
                self.event_num,
                self.led_color,
                self.condition_label,
                self.frames_received,
                sample_rate,
                self.config
            )
            with open(json_path, 'w') as f:
                json.dump(sidecar, f, indent=2)
        except Exception as e:
            # Log warning but continue (binary is more important)
            print(f"    ⚠ Sidecar JSON save failed: {e}")
        
        return True


async def stream_event(
    event: ExperimentEvent,
    experiment_start: float,
    status_cb: Optional[Callable] = None,
    stop_flag: Optional[threading.Event] = None
) -> ExperimentEvent:
    """
    Stream frames for a single event.
    
    This function connects to WebSocket and receives all frames for the event.
    
    Args:
        event: ExperimentEvent instance
        experiment_start: Timestamp of experiment start
        status_cb: Optional callback for status updates
        stop_flag: Optional threading.Event to check for stop requests
        
    Returns:
        The same ExperimentEvent instance with updated state
    """
    # Small delay for route activation
    await asyncio.sleep(0.3)
    
    # Receive frames
    try:
        async with websockets.connect(event.client.ws_url) as websocket:
            await event.receive_frames(websocket, experiment_start, status_cb, stop_flag)
    except Exception as e:
        event.error_message = f"WebSocket connection error: {e}"
        t_offset = time.time() - experiment_start
        print(f"    [E{event.event_num}] T+{t_offset:.1f}s | WS ERROR: {event.error_message}")
    
    return event


def background_save(event: ExperimentEvent, output_folder: str) -> tuple:
    """
    Background save function for ThreadPoolExecutor.
    
    Args:
        event: ExperimentEvent instance
        output_folder: Path to output folder
        
    Returns:
        Tuple of (event_num, success, filename, error_message)
    """
    success = event.save_binary(output_folder)
    return (event.event_num, success, event.filename, event.error_message)


async def run_config_experiment(
    config: dict,
    schedule: List[tuple],
    client: HardwareClient,
    skip_prompts: bool = False,
    status_cb: Optional[Callable] = None,
    stop_flag: Optional[threading.Event] = None,
    skip_hardware_setup: bool = False
) -> bool:
    """
    Run config-driven experiment with optional hooks for web UI.
    
    Args:
        config: Configuration dictionary
        schedule: List of (led_color, condition_label) tuples
        client: HardwareClient instance
        skip_prompts: If True, skip user confirmation prompts (for web mode)
        status_cb: Optional callback for status updates (for web UI)
        stop_flag: Optional threading.Event to check for stop requests
        skip_hardware_setup: If True, skip profile loading and register dump
                            (useful when caller already handled these steps)
        
    Returns:
        True if successful (including clean stop), False on error
    """
    # Extract values from config
    num_events = config['experiment']['num_events']
    inter_event_delay = config['experiment'].get('inter_event_delay', DEFAULT_INTER_EVENT_DELAY)
    profile_path = config['experiment'].get('profile_path', None)
    
    # ============ HARDWARE SETUP (Profile Loading + Register Dump) ============
    if not skip_hardware_setup:
        # Step 1: Load and apply profile if specified
        if profile_path:
            print(f"\n[HARDWARE SETUP] Loading ADS1299 profile...")
            if status_cb:
                try:
                    status_cb({"type": "setup", "step": "profile_loading", "path": profile_path})
                except Exception:
                    pass
            
            if not client.load_and_apply_profile(profile_path):
                print(f"\n✗ Profile loading failed: {profile_path}")
                if status_cb:
                    try:
                        status_cb({"type": "error", "message": f"Profile loading failed: {profile_path}"})
                    except Exception:
                        pass
                return False
            
            # Brief delay after profile application for hardware to settle
            await asyncio.sleep(0.5)
            print(f"    ✓ Profile applied successfully")
        else:
            print(f"\n[HARDWARE SETUP] No profile specified - using current ADS1299 configuration")
        
        # Step 2: Dump and verify ADS1299 registers
        print(f"\n[HARDWARE SETUP] Verifying ADS1299 configuration...")
        if status_cb:
            try:
                status_cb({"type": "setup", "step": "register_dump"})
            except Exception:
                pass
        
        registers = client.dump_and_verify_registers()
        if not registers:
            print(f"\n✗ Could not read ADS1299 registers")
            if status_cb:
                try:
                    status_cb({"type": "error", "message": "Could not read ADS1299 registers"})
                except Exception:
                    pass
            return False
        
        # Emit register dump info to status callback
        if status_cb:
            try:
                status_cb({
                    "type": "setup",
                    "step": "register_dump_complete",
                    "sample_rate": client.get_sample_rate(),
                    "registers": registers
                })
            except Exception:
                pass
        
        print(f"    ✓ ADS1299 registers verified (Sample rate: {client.get_sample_rate()} SPS)")
    # =========================================================================
    
    # Compute output folder based on meta section
    output_folder = compute_output_folder(config)
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ Output folder ready: {output_folder}/")
    
    # Emit running state
    if status_cb:
        try:
            status_cb({"type": "state", "status": "running"})
        except Exception:
            pass
    
    save_executor = ThreadPoolExecutor(max_workers=MAX_SAVE_WORKERS)
    pending_saves = []
    results = []
    event_durations = []
    experiment_start = time.time()
    
    # Run events from schedule
    for idx, (led_color, condition_label) in enumerate(schedule, 1):
        # Check stop flag at loop start
        if stop_flag and stop_flag.is_set():
            print(f"\n⚠ Experiment stopped cleanly after {idx-1} events")
            if status_cb:
                try:
                    status_cb({"type": "state", "status": "stopped"})
                except Exception:
                    pass
            # Clean stop = success
            save_executor.shutdown(wait=True)
            return True
        
        event_start = time.time()
        t_offset = event_start - experiment_start
        
        event = ExperimentEvent(idx, led_color, condition_label, config, client)
        
        print(f"\n{'='*60}")
        print(f"EVENT {idx}/{num_events} | LED: {led_color} | Condition: {condition_label} | T+{t_offset:.1f}s")
        print(f"{'='*60}")
        
        # Emit event start
        if status_cb:
            try:
                status_cb({
                    "type": "event_start",
                    "event": idx,
                    "total": num_events,
                    "led_color": led_color,
                    "condition": condition_label,
                    "expected_frames": config['routine']['frames_per_event']
                })
            except Exception:
                pass
        
        print(f"  [1] Preparing routine...")
        if not await event.prepare_routine():
            print(f"      ✗ {event.error_message}")
            results.append(event)
            if idx < num_events:
                await asyncio.sleep(inter_event_delay)
            continue
        print(f"      ✓ Prepared")
        
        print(f"  [2] Executing routine...")
        if not await event.execute_routine():
            print(f"      ✗ {event.error_message}")
            results.append(event)
            if idx < num_events:
                await asyncio.sleep(inter_event_delay)
            continue
        print(f"      ✓ Started streaming")
        
        print(f"  [3] Receiving frames...")
        await stream_event(event, experiment_start, status_cb, stop_flag)
        
        stream_duration = event.end_time - event.start_time if event.start_time and event.end_time else 0
        event_durations.append(stream_duration)
        
        if event.success:
            expected_frames = config['routine']['frames_per_event']
            print(f"      ✓ Received {event.frames_received}/{expected_frames} frames in {stream_duration:.2f}s")
            
            # Emit event done
            if status_cb:
                try:
                    status_cb({
                        "type": "event_done",
                        "event": idx,
                        "success": True,
                        "error": None
                    })
                except Exception:
                    pass
            
            print(f"  [4] Queuing background save...")
            future = save_executor.submit(background_save, event, output_folder)
            pending_saves.append((event, future))
            print(f"      ✓ Save queued")
        else:
            print(f"      ✗ {event.error_message}")
            # Emit event failed
            if status_cb:
                try:
                    status_cb({
                        "type": "event_done",
                        "event": idx,
                        "success": False,
                        "error": event.error_message
                    })
                except Exception:
                    pass
        
        results.append(event)
        
        if idx < num_events:
            await asyncio.sleep(inter_event_delay)
    
    # Wait for saves
    print("\n" + "="*60)
    print(f"Waiting for {len(pending_saves)} background saves...")
    print("="*60)
    
    save_successes = 0
    save_failures = 0
    saved_files = []
    
    for event, future in pending_saves:
        try:
            event_num, success, filename, error_msg = future.result(timeout=30)
            if success:
                save_successes += 1
                saved_files.append(filename)
                print(f"  ✓ Event {event_num}: {filename}")
                
                # Emit file saved
                if status_cb:
                    try:
                        status_cb({
                            "type": "file_saved",
                            "event": event_num,
                            "filename": filename,
                            "success": True
                        })
                    except Exception:
                        pass
            else:
                save_failures += 1
                print(f"  ✗ Event {event_num}: {error_msg}")
                
                # Emit save failure
                if status_cb:
                    try:
                        status_cb({
                            "type": "file_saved",
                            "event": event_num,
                            "filename": None,
                            "success": False
                        })
                    except Exception:
                        pass
        except Exception as e:
            save_failures += 1
            print(f"  ✗ Event {event.event_num}: {e}")
    
    save_executor.shutdown(wait=True)
    
    # Summary
    successes = sum(1 for e in results if e.success)
    failures = sum(1 for e in results if not e.success)
    total_duration = time.time() - experiment_start
    avg_stream = sum(event_durations) / len(event_durations) if event_durations else 0
    
    print("\n" + "="*70)
    print("=== EXPERIMENT SUMMARY ===")
    print("="*70)
    print(f"\nTotal events:    {num_events}")
    print(f"Successes:       {successes}")
    print(f"Failures:        {failures}")
    print(f"Saved files:     {save_successes}")
    print(f"Total time:      {total_duration:.1f}s")
    print(f"Avg stream time: {avg_stream:.2f}s")
    print(f"Output:          {output_folder}/")
    
    if saved_files:
        print(f"\n📁 Saved files:")
        for f in saved_files:
            print(f"    ✓ {f}")
    
    # Emit summary
    if status_cb:
        try:
            status_cb({
                "type": "summary",
                "stream_successes": successes,
                "stream_failures": failures,
                "save_successes": save_successes,
                "save_failures": save_failures,
                "output_folder": output_folder
            })
        except Exception:
            pass
    
    print("\n" + "="*70)
    if failures == 0 and save_failures == 0:
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
        return True
    else:
        print(f"⚠️ COMPLETED WITH {failures + save_failures} FAILURE(S)")
        return False


async def run_experiment(
    num_events: int,
    client: HardwareClient,
    profile_path: Optional[str] = None,
    output_folder: str = "experiments"
) -> bool:
    """
    Run legacy experiment with SEQUENTIAL streaming and background saves.
    
    This is the legacy mode for simple experiments without full config.
    
    Args:
        num_events: Number of events to run
        client: HardwareClient instance
        profile_path: Optional path to profile file
        output_folder: Output folder for binary files
        
    Returns:
        True if all events successful, False otherwise
    """
    print("="*70)
    print("=== ADS1299 EXPERIMENT RUNNER (LEGACY MODE) ===")
    print("="*70)
    print(f"Device:              {client.ip}")
    print(f"Events:              {num_events}")
    print(f"Frames per event:    {DEFAULT_FRAMES_PER_EVENT}")
    print(f"Samples per event:   {DEFAULT_FRAMES_PER_EVENT * SAMPLES_PER_FRAME}")
    print(f"Sound duration:      {DEFAULT_SOUND_DURATION_US / 1000000:.1f} seconds")
    print(f"Inter-event delay:   {DEFAULT_INTER_EVENT_DELAY}s (hardware reset)")
    print(f"Output folder:       {output_folder}/")
    print(f"Profile:             {profile_path if profile_path else 'default (no change)'}")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ Output folder ready: {output_folder}/")
    
    # Load profile if specified
    if profile_path:
        if not client.load_and_apply_profile(profile_path):
            print("\n✗ EXPERIMENT ABORTED: Profile loading failed")
            return False
        await asyncio.sleep(0.5)
    
    # Dump and verify registers
    print("\n[CONFIG] Verifying ADS1299 configuration...")
    registers = client.dump_and_verify_registers()
    if not registers:
        print("\n✗ EXPERIMENT ABORTED: Could not read registers")
        return False
    
    # Build a simple config for legacy mode
    config = {
        'experiment': {
            'num_events': num_events,
            'output_folder': output_folder
        },
        'routine': {
            'frames_per_event': DEFAULT_FRAMES_PER_EVENT,
            'sound_duration_us': DEFAULT_SOUND_DURATION_US,
            'led_enabled': True
        },
        'conditions': {
            'mapping': {c: c for c in VALID_LED_COLORS}
        }
    }
    
    # Build schedule with random colors
    schedule = [(random.choice(VALID_LED_COLORS), "LEGACY") for _ in range(num_events)]
    
    # Run using config-driven function
    return await run_config_experiment(
        config=config,
        schedule=schedule,
        client=client,
        skip_prompts=False,
        skip_hardware_setup=True  # Already done above
    )
