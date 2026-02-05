#!/usr/bin/env python3
"""
Web UI for ADS1299 Experiment Runner

Provides a browser-based interface for configuring and running experiments.
Complements the CLI mode with visual feedback and form-based configuration.

Usage:
    python web_ui.py
    
    (Browser will open automatically to http://localhost:8000)
"""

import asyncio
import json
import os
import threading
import queue
import webbrowser
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Resolve script directory for robust relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

# Import from existing experiment runner
from run_experiment import (
    load_config,
    validate_experiment_config,
    generate_event_schedule,
    run_config_experiment,
    setup_random_seed,
    load_and_apply_profile,
    dump_and_verify_registers,
    compute_output_folder,
)

# Initialize FastAPI
app = FastAPI(title="ADS1299 Experiment Runner")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=str(SCRIPT_DIR / "static")), name="static")

# Global state (thread-safe)
experiment_lock = threading.Lock()
experiment_running = False
experiment_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
status_queue = queue.Queue()

# Stored configuration
stored_config: Optional[dict] = None
stored_schedule: Optional[list] = None

# Active WebSocket connections
active_connections: List[WebSocket] = []


def status_callback(event_dict: dict):
    """
    Called from experiment thread to send status updates.
    Thread-safe: puts events in queue for async broadcaster.
    """
    status_queue.put(event_dict)


async def broadcast_status(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    disconnected = []
    for websocket in active_connections:
        try:
            await websocket.send_json(message)
        except Exception:
            disconnected.append(websocket)
    
    # Clean up disconnected clients
    for websocket in disconnected:
        if websocket in active_connections:
            active_connections.remove(websocket)


async def status_broadcaster():
    """Background task that drains status queue and broadcasts"""
    while True:
        try:
            # Non-blocking check for status updates
            event = status_queue.get_nowait()
            await broadcast_status(event)
        except queue.Empty:
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():
    """Start background broadcaster"""
    asyncio.create_task(status_broadcaster())


@app.get("/")
async def root():
    """Serve main UI"""
    return FileResponse(str(SCRIPT_DIR / "static" / "index.html"))


@app.get("/api/status")
async def get_status():
    """Get current experiment status"""
    return {
        "running": experiment_running,
        "config_loaded": stored_config is not None,
        "schedule_ready": stored_schedule is not None,
    }


@app.post("/api/load-config")
async def load_config_endpoint(request: dict):
    """Load and validate configuration from file"""
    global stored_config, stored_schedule
    
    config_path = request.get("config_path", "config.json")
    
    try:
        config = load_config(config_path)
        validate_experiment_config(config)
        
        stored_config = config
        stored_schedule = None  # Reset schedule when config changes
        
        return {
            "success": True,
            "config": config,
            "message": f"Config loaded from {config_path}"
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/save-config")
async def save_config_endpoint(request: dict):
    """Save configuration to config.json"""
    global stored_config, stored_schedule
    
    try:
        config = request.get("config")
        if not config:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No config provided"}
            )
        
        # Validate before saving
        validate_experiment_config(config)
        
        # Write to disk
        config_path = "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        stored_config = config
        stored_schedule = None  # Reset schedule when config changes
        
        return {
            "success": True,
            "message": f"Config saved to {config_path}"
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/generate-schedule")
async def generate_schedule_endpoint():
    """Generate event schedule from loaded config"""
    global stored_schedule
    
    if not stored_config:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No config loaded"}
        )
    
    try:
        # Set random seed if specified
        setup_random_seed(stored_config)
        
        # Generate schedule
        schedule = generate_event_schedule(
            num_events=stored_config['experiment']['num_events'],
            condition_mapping=stored_config['conditions']['mapping'],
            enforce_equal_count=stored_config['conditions']['enforce_equal_condition_count']
        )
        
        stored_schedule = schedule
        
        # Calculate statistics
        label_counts = {}
        for led_color, condition_label in schedule:
            label_counts[condition_label] = label_counts.get(condition_label, 0) + 1
        
        return {
            "success": True,
            "schedule_length": len(schedule),
            "counts": label_counts,
            "preview": schedule[:10]  # First 10 events
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/run-experiment")
async def run_experiment_endpoint():
    """Start experiment in background thread"""
    global experiment_running, experiment_thread
    
    # Check if already running
    if experiment_running:
        return JSONResponse(
            status_code=409,
            content={"success": False, "error": "Experiment already running"}
        )
    
    # Check prerequisites
    if not stored_config:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No config loaded"}
        )
    
    if not stored_schedule:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Schedule not generated"}
        )
    
    # Set running flag with lock
    with experiment_lock:
        if experiment_running:
            return JSONResponse(
                status_code=409,
                content={"success": False, "error": "Experiment already running"}
            )
        experiment_running = True
    
    # Clear stop flag
    stop_event.clear()
    
    # Send initial status
    await broadcast_status({"type": "state", "status": "starting"})
    
    def run_experiment_wrapper():
        """Wrapper that runs experiment and handles cleanup"""
        global experiment_running
        try:
            # Run experiment with hooks
            asyncio.run(run_config_experiment(
                config=stored_config,
                schedule=stored_schedule,
                skip_prompts=True,  # No input() prompts in web mode
                status_cb=status_callback,  # Send updates via queue
                stop_flag=stop_event  # Check for stop requests
            ))
        except Exception as e:
            # Send error to UI
            status_callback({
                "type": "state",
                "status": "error",
                "error": str(e)
            })
        finally:
            experiment_running = False
            status_callback({"type": "state", "status": "finished"})
    
    # Start experiment in background thread
    experiment_thread = threading.Thread(target=run_experiment_wrapper, daemon=True)
    experiment_thread.start()
    
    return {"success": True, "message": "Experiment started"}


@app.post("/api/stop-experiment")
async def stop_experiment_endpoint():
    """Stop experiment after current event"""
    if not experiment_running:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No experiment running"}
        )
    
    stop_event.set()
    await broadcast_status({
        "type": "state",
        "status": "stopping",
        "message": "Will stop after current event finishes"
    })
    
    return {
        "success": True,
        "message": "Stop requested. Will stop after current event."
    }


@app.get("/api/detect-sample-rate")
async def detect_sample_rate():
    """Auto-detect sample rate from ESP32 register dump"""
    if not stored_config:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No config loaded"}
        )
    
    esp_ip = stored_config['esp32']['ip']
    
    try:
        response = requests.get(
            f"http://{esp_ip}/ads1299/diagnostics/registers-dump",
            timeout=5
        )
        if response.status_code == 200:
            registers = response.json()
            config1_hex = registers.get('CONFIG1', '0x00')
            config1_val = int(config1_hex, 16) if isinstance(config1_hex, str) else config1_hex
            odr_bits = config1_val & 0x07
            
            # Sample rate lookup
            rate_map = {0: 16000, 1: 8000, 2: 4000, 3: 2000, 4: 1000, 5: 500, 6: 250}
            sample_rate = rate_map.get(odr_bits, 16000)
            
            return {
                "success": True,
                "sample_rate_sps": sample_rate,
                "config1": config1_hex
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"HTTP {response.status_code}"}
            )
    except requests.exceptions.RequestException as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/list-files")
async def list_files_endpoint():
    """List .bin and .json files from computed output folder"""
    if not stored_config:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No config loaded"}
        )
    
    # Import helper from run_experiment
    from run_experiment import compute_output_folder
    output_folder = compute_output_folder(stored_config)
    
    if not os.path.exists(output_folder):
        return {
            "success": True,
            "files": [],
            "folder": output_folder,
            "total_files": 0,
            "total_size": 0
        }
    
    files = []
    try:
        for filename in os.listdir(output_folder):
            if filename.endswith('.bin') or filename.endswith('.json'):
                filepath = os.path.join(output_folder, filename)
                stat = os.stat(filepath)
                files.append({
                    "name": filename,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": "binary" if filename.endswith('.bin') else "metadata"
                })
        
        # Sort by modified time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return {
            "success": True,
            "files": files,
            "folder": output_folder,
            "total_files": len(files),
            "total_size": sum(f['size'] for f in files)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/list-profiles")
async def list_profiles_endpoint():
    """List available .json profile files from profiles/ directory"""
    try:
        profiles_dir = os.path.join(os.path.dirname(__file__), 'profiles')
        profiles = []
        if os.path.exists(profiles_dir):
            profiles = sorted([f for f in os.listdir(profiles_dir) if f.endswith('.json')])
        return {"success": True, "profiles": profiles, "profiles_dir": "profiles/"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/profile-content")
async def get_profile_content(filename: str):
    """Get content of a profile file from profiles/ directory"""
    try:
        profiles_dir = os.path.join(os.path.dirname(__file__), 'profiles')
        filepath = os.path.join(profiles_dir, filename)
        
        # Security: ensure filename doesn't escape profiles directory
        if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid filename"}
            )
        
        if not os.path.exists(filepath):
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Profile not found: {filename}"}
            )
        
        with open(filepath, 'r') as f:
            content = json.load(f)
        
        return {
            "success": True,
            "filename": filename,
            "content": content
        }
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Invalid JSON: {e}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/run-all-profiles")
async def run_all_profiles_endpoint(request: dict = None):
    """
    Run validation streams across all profiles sequentially.
    
    Each profile is:
    1. Loaded and applied to ESP32
    2. Run with a 5-second validation stream
    3. Saved with profile name in filename
    
    WebSocket updates are sent for progress tracking.
    """
    global experiment_running, experiment_thread
    
    # Check if already running
    if experiment_running:
        return JSONResponse(
            status_code=409,
            content={"success": False, "error": "Experiment already running"}
        )
    
    # Get ESP32 IP from request or use default
    esp_ip = "node.local"
    if request and request.get("esp_ip"):
        esp_ip = request["esp_ip"]
    elif stored_config and stored_config.get("esp32", {}).get("ip"):
        esp_ip = stored_config["esp32"]["ip"]
    
    firmware = request.get("firmware", "FW2") if request else "FW2"
    output_folder = request.get("output_folder", "experiments") if request else "experiments"
    
    # List all profiles
    profiles_dir = os.path.join(os.path.dirname(__file__), 'profiles')
    if not os.path.exists(profiles_dir):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No profiles directory found"}
        )
    
    profiles = sorted([f for f in os.listdir(profiles_dir) if f.endswith('.json')])
    if not profiles:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No profile files found in profiles/"}
        )
    
    # Set running flag with lock
    with experiment_lock:
        if experiment_running:
            return JSONResponse(
                status_code=409,
                content={"success": False, "error": "Experiment already running"}
            )
        experiment_running = True
    
    # Clear stop flag
    stop_event.clear()
    
    # Send initial status
    await broadcast_status({
        "type": "all_profiles_start",
        "total_profiles": len(profiles),
        "profiles": profiles
    })
    
    def run_all_profiles_wrapper():
        """Run validation for each profile sequentially"""
        global experiment_running
        import run_experiment
        
        # Set globals in run_experiment module for API calls
        run_experiment.ESP32_IP = esp_ip
        run_experiment.HTTP_URL = f"http://{esp_ip}"
        run_experiment.WS_URL = f"ws://{esp_ip}/ws"
        
        results = []
        successful = 0
        failed = 0
        
        try:
            for idx, profile_name in enumerate(profiles, 1):
                # Check for stop request
                if stop_event.is_set():
                    status_callback({
                        "type": "all_profiles_stopped",
                        "completed": idx - 1,
                        "total": len(profiles)
                    })
                    break
                
                profile_path = os.path.join(profiles_dir, profile_name)
                profile_basename = profile_name.replace('.json', '')
                
                # Send progress update
                status_callback({
                    "type": "profile_progress",
                    "current": idx,
                    "total": len(profiles),
                    "profile": profile_name,
                    "status": "loading"
                })
                
                # Step 1: Load and apply profile
                try:
                    if not load_and_apply_profile(profile_path):
                        status_callback({
                            "type": "profile_progress",
                            "current": idx,
                            "total": len(profiles),
                            "profile": profile_name,
                            "status": "failed",
                            "error": "Profile loading failed"
                        })
                        results.append({"profile": profile_name, "success": False, "error": "Profile loading failed"})
                        failed += 1
                        continue
                except Exception as e:
                    status_callback({
                        "type": "profile_progress",
                        "current": idx,
                        "total": len(profiles),
                        "profile": profile_name,
                        "status": "failed",
                        "error": str(e)
                    })
                    results.append({"profile": profile_name, "success": False, "error": str(e)})
                    failed += 1
                    continue
                
                # Brief delay for hardware to settle
                import time
                time.sleep(0.5)
                
                # Step 2: Dump registers to get sample rate
                registers = dump_and_verify_registers()
                sample_rate = run_experiment.SAMPLE_RATE or 16000
                
                # Calculate frames for 5 seconds
                # frames = (sample_rate * duration) / samples_per_frame
                frames = int((sample_rate * 5) / 50)  # 50 samples per frame
                
                status_callback({
                    "type": "profile_progress",
                    "current": idx,
                    "total": len(profiles),
                    "profile": profile_name,
                    "status": "streaming",
                    "frames": frames,
                    "sample_rate": sample_rate
                })
                
                # Step 3: Build validation config with profile name in filename
                config = {
                    "esp32": {"ip": esp_ip},
                    "experiment": {
                        "num_events": 1,
                        "random_seed": None,
                        "profile_path": None,  # Already applied
                        "output_folder": output_folder,
                        "inter_event_delay": 0.5
                    },
                    "routine": {
                        "frames_per_event": frames,
                        "sound_duration_us": 0,
                        "led_enabled": False
                    },
                    "conditions": {
                        "enforce_equal_condition_count": False,
                        "mapping": {"RED": "VAL", "BLUE": "VAL", "GREEN": "VAL", "YELLOW": "VAL"}
                    },
                    "meta": {
                        "mode": "validation",
                        "firmware": firmware,
                        "board": "R2",
                        "validation_scenario": "profile_test",
                        "profile_validation_name": profile_basename  # Used in filename
                    }
                }
                
                # Single-event schedule
                schedule = [("RED", "VAL")]
                
                # Step 4: Run the experiment
                try:
                    success = asyncio.run(run_config_experiment(
                        config=config,
                        schedule=schedule,
                        skip_prompts=True,
                        status_cb=status_callback,
                        stop_flag=stop_event,
                        skip_hardware_setup=True  # Profile already applied
                    ))
                    
                    if success:
                        status_callback({
                            "type": "profile_progress",
                            "current": idx,
                            "total": len(profiles),
                            "profile": profile_name,
                            "status": "completed"
                        })
                        results.append({"profile": profile_name, "success": True})
                        successful += 1
                    else:
                        status_callback({
                            "type": "profile_progress",
                            "current": idx,
                            "total": len(profiles),
                            "profile": profile_name,
                            "status": "failed",
                            "error": "Streaming failed"
                        })
                        results.append({"profile": profile_name, "success": False, "error": "Streaming failed"})
                        failed += 1
                        
                except Exception as e:
                    status_callback({
                        "type": "profile_progress",
                        "current": idx,
                        "total": len(profiles),
                        "profile": profile_name,
                        "status": "failed",
                        "error": str(e)
                    })
                    results.append({"profile": profile_name, "success": False, "error": str(e)})
                    failed += 1
                
                # Brief delay between profiles
                if idx < len(profiles):
                    time.sleep(1.0)
            
            # Send final summary
            status_callback({
                "type": "all_profiles_complete",
                "total": len(profiles),
                "successful": successful,
                "failed": failed,
                "results": results
            })
            
        except Exception as e:
            status_callback({
                "type": "state",
                "status": "error",
                "error": str(e)
            })
        finally:
            experiment_running = False
            status_callback({"type": "state", "status": "finished"})
    
    # Start in background thread
    experiment_thread = threading.Thread(target=run_all_profiles_wrapper, daemon=True)
    experiment_thread.start()
    
    return {
        "success": True,
        "message": f"Started validation for {len(profiles)} profiles",
        "profiles": profiles
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live status updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Keep connection alive (no messages required from client)
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)


def open_browser():
    """Open default browser to the UI"""
    webbrowser.open('http://localhost:8000')


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 ADS1299 Experiment Runner - Web UI")
    print("="*60)
    print("\nStarting server on http://localhost:8000")
    print("Browser will open automatically...")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Open browser after 1.5 seconds
    threading.Timer(1.5, open_browser).start()
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
