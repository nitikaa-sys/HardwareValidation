#!/usr/bin/env python3
"""
Web UI for ADS1299 Experiment Runner - Thin Endpoints Only

All business logic delegated to acquisition modules.
API routes and response formats unchanged from original.

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
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests

# Module imports
from acquisition import (
    HardwareClient,
    load_config,
    validate_experiment_config,
    setup_random_seed,
    generate_event_schedule,
    run_config_experiment,
    get_profiles_dir,
    PACKAGE_DIR
)
from acquisition.constants import (
    SAMPLES_PER_FRAME,
    DEFAULT_ESP32_IP,
    SAMPLE_RATE_MAP
)
from acquisition.naming import compute_output_folder

# Resolve script directory for robust relative paths
SCRIPT_DIR = PACKAGE_DIR

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
    """Called from experiment thread to send status updates."""
    status_queue.put(event_dict)


async def broadcast_status(message: dict):
    """Broadcast message to all connected WebSocket clients."""
    disconnected = []
    for websocket in active_connections:
        try:
            await websocket.send_json(message)
        except Exception:
            disconnected.append(websocket)
    
    for websocket in disconnected:
        if websocket in active_connections:
            active_connections.remove(websocket)


async def status_broadcaster():
    """Background task that drains status queue and broadcasts."""
    while True:
        try:
            event = status_queue.get_nowait()
            await broadcast_status(event)
        except queue.Empty:
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():
    """Start background broadcaster."""
    asyncio.create_task(status_broadcaster())


# =============================================================================
# API ENDPOINTS - Thin wrappers around module functions
# =============================================================================

@app.get("/")
async def root():
    """Serve main UI."""
    return FileResponse(str(SCRIPT_DIR / "static" / "index.html"))


@app.get("/api/status")
async def get_status():
    """Get current experiment status."""
    return {
        "running": experiment_running,
        "config_loaded": stored_config is not None,
        "schedule_ready": stored_schedule is not None,
    }


@app.post("/api/load-config")
async def load_config_endpoint(request: dict):
    """Load and validate configuration from file."""
    global stored_config, stored_schedule
    
    config_path = request.get("config_path", "config.json")
    
    try:
        config = load_config(config_path)
        validate_experiment_config(config)
        
        stored_config = config
        stored_schedule = None
        
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
    """Save configuration to config.json."""
    global stored_config, stored_schedule
    
    try:
        config = request.get("config")
        if not config:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No config provided"}
            )
        
        validate_experiment_config(config)
        
        config_path = str(SCRIPT_DIR / "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        stored_config = config
        stored_schedule = None
        
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
    """Generate event schedule from loaded config."""
    global stored_schedule
    
    if not stored_config:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No config loaded"}
        )
    
    try:
        setup_random_seed(stored_config)
        
        schedule = generate_event_schedule(
            num_events=stored_config['experiment']['num_events'],
            condition_mapping=stored_config['conditions']['mapping'],
            enforce_equal_count=stored_config['conditions']['enforce_equal_condition_count']
        )
        
        stored_schedule = schedule
        
        label_counts = {}
        for led_color, condition_label in schedule:
            label_counts[condition_label] = label_counts.get(condition_label, 0) + 1
        
        return {
            "success": True,
            "schedule_length": len(schedule),
            "counts": label_counts,
            "preview": schedule[:10]
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/run-experiment")
async def run_experiment_endpoint():
    """Start experiment in background thread."""
    global experiment_running, experiment_thread
    
    if experiment_running:
        return JSONResponse(
            status_code=409,
            content={"success": False, "error": "Experiment already running"}
        )
    
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
    
    with experiment_lock:
        if experiment_running:
            return JSONResponse(
                status_code=409,
                content={"success": False, "error": "Experiment already running"}
            )
        experiment_running = True
    
    stop_event.clear()
    await broadcast_status({"type": "state", "status": "starting"})
    
    def run_experiment_wrapper():
        global experiment_running
        try:
            client = HardwareClient(stored_config['esp32']['ip'])
            asyncio.run(run_config_experiment(
                config=stored_config,
                schedule=stored_schedule,
                client=client,
                skip_prompts=True,
                status_cb=status_callback,
                stop_flag=stop_event
            ))
        except Exception as e:
            status_callback({
                "type": "state",
                "status": "error",
                "error": str(e)
            })
        finally:
            experiment_running = False
            status_callback({"type": "state", "status": "finished"})
    
    experiment_thread = threading.Thread(target=run_experiment_wrapper, daemon=True)
    experiment_thread.start()
    
    return {"success": True, "message": "Experiment started"}


@app.post("/api/stop-experiment")
async def stop_experiment_endpoint():
    """Stop experiment after current event."""
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
    """Auto-detect sample rate from ESP32 register dump."""
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
            sample_rate = SAMPLE_RATE_MAP.get(odr_bits, 16000)
            
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
    """List .bin and .json files from computed output folder."""
    if not stored_config:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No config loaded"}
        )
    
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
    """List available .json profile files from profiles/ directory."""
    try:
        profiles_dir = get_profiles_dir()
        profiles = []
        if profiles_dir.exists():
            profiles = sorted([f for f in os.listdir(profiles_dir) if f.endswith('.json')])
        return {"success": True, "profiles": profiles, "profiles_dir": "profiles/"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/profile-content")
async def get_profile_content(filename: str):
    """Get content of a profile file from profiles/ directory."""
    try:
        profiles_dir = get_profiles_dir()
        filepath = profiles_dir / filename
        
        if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid filename"}
            )
        
        if not filepath.exists():
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
    """Run validation streams across all profiles sequentially."""
    global experiment_running, experiment_thread
    
    if experiment_running:
        return JSONResponse(
            status_code=409,
            content={"success": False, "error": "Experiment already running"}
        )
    
    esp_ip = DEFAULT_ESP32_IP
    if request and request.get("esp_ip"):
        esp_ip = request["esp_ip"]
    elif stored_config and stored_config.get("esp32", {}).get("ip"):
        esp_ip = stored_config["esp32"]["ip"]
    
    firmware = request.get("firmware", "FW2") if request else "FW2"
    output_folder = request.get("output_folder", "experiments") if request else "experiments"
    
    profiles_dir = get_profiles_dir()
    if not profiles_dir.exists():
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
    
    with experiment_lock:
        if experiment_running:
            return JSONResponse(
                status_code=409,
                content={"success": False, "error": "Experiment already running"}
            )
        experiment_running = True
    
    stop_event.clear()
    
    await broadcast_status({
        "type": "all_profiles_start",
        "total_profiles": len(profiles),
        "profiles": profiles
    })
    
    def run_all_profiles_wrapper():
        global experiment_running
        
        client = HardwareClient(esp_ip)
        results = []
        successful = 0
        failed = 0
        
        try:
            for idx, profile_name in enumerate(profiles, 1):
                if stop_event.is_set():
                    status_callback({
                        "type": "all_profiles_stopped",
                        "completed": idx - 1,
                        "total": len(profiles)
                    })
                    break
                
                profile_path = str(profiles_dir / profile_name)
                profile_basename = profile_name.replace('.json', '')
                
                status_callback({
                    "type": "profile_progress",
                    "current": idx,
                    "total": len(profiles),
                    "profile": profile_name,
                    "status": "loading"
                })
                
                try:
                    if not client.load_and_apply_profile(profile_path):
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
                
                time.sleep(0.5)
                
                registers = client.dump_and_verify_registers()
                sample_rate = client.get_sample_rate()
                
                frames = int((sample_rate * 5) / SAMPLES_PER_FRAME)
                
                status_callback({
                    "type": "profile_progress",
                    "current": idx,
                    "total": len(profiles),
                    "profile": profile_name,
                    "status": "streaming",
                    "frames": frames,
                    "sample_rate": sample_rate
                })
                
                config = {
                    "esp32": {"ip": esp_ip},
                    "experiment": {
                        "num_events": 1,
                        "random_seed": None,
                        "profile_path": None,
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
                        "profile_validation_name": profile_basename
                    }
                }
                
                schedule = [("RED", "VAL")]
                
                try:
                    success = asyncio.run(run_config_experiment(
                        config=config,
                        schedule=schedule,
                        client=client,
                        skip_prompts=True,
                        status_cb=status_callback,
                        stop_flag=stop_event,
                        skip_hardware_setup=True
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
                
                if idx < len(profiles):
                    time.sleep(1.0)
            
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
    
    experiment_thread = threading.Thread(target=run_all_profiles_wrapper, daemon=True)
    experiment_thread.start()
    
    return {
        "success": True,
        "message": f"Started validation for {len(profiles)} profiles",
        "profiles": profiles
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live status updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)


def open_browser():
    """Open default browser to the UI."""
    webbrowser.open('http://localhost:8000')


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADS1299 Experiment Runner - Web UI")
    print("="*60)
    print("\nStarting server on http://localhost:8000")
    print("Browser will open automatically...")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    threading.Timer(1.5, open_browser).start()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
