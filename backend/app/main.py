from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading
import cv2
import numpy as np
import time
import os
import io
import csv
from datetime import datetime
from app.cv_model.detector import OccupancyDetector, ApplianceDetector
from app.utils.privacy import apply_ghost_mode

# --- GLOBAL STATE ---
# We store the latest processed CV results in memory so the API can serve it instantly.
# We include one LIVE room hooked up to your camera, and two MOCK rooms for dashboard aesthetics.
global_frame = None
camera_status = {"connected": False, "message": "Initializing..."}
history_log = []  # Stores timestamped snapshots of room state
MAX_HISTORY = 500  # Keep last 500 entries

ROOMS_STATE = [
    {
        "id": "Room 101 (Live Cam)",
        "person_count": 0,
        "appliance_state": "OFF",
        "appliance_count": 0,
        "alert": False,
        "energy_saved_kwh": 0.0
    },
    {
        "id": "Room 102 (Class)",
        "person_count": 34,
        "appliance_state": "ON",
        "appliance_count": 6,
        "alert": False,
        "energy_saved_kwh": 14.5
    },
    {
        "id": "Room 103 (Lab)",
        "person_count": 0,
        "appliance_state": "OFF",
        "appliance_count": 0,
        "alert": False,
        "energy_saved_kwh": 8.1
    }
]

def generate_status_frame(message="NO CAMERA", sub_message="Waiting for video source..."):
    """Generate a diagnostic frame when no camera is available."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Dark blue background
    frame[:] = (30, 15, 5)
    # Draw border
    cv2.rectangle(frame, (10, 10), (630, 470), (255, 229, 0), 1)
    # Main text
    cv2.putText(frame, message, (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 229, 0), 2)
    # Sub text
    cv2.putText(frame, sub_message, (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # Timestamp
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"WATT-WATCH // {timestamp}", (180, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    return frame


def vision_processing_loop():
    """
    This function runs continuously in the background parsing the camera.
    """
    global global_frame, camera_status
    print("[BACKGROUND] Starting Vision Processing Thread...")
    
    try:
        detector = OccupancyDetector()
        appliance_detector = ApplianceDetector()
    except Exception as e:
        print("[WARNING] CV Models failed to initialize:", e)
        camera_status = {"connected": False, "message": f"Model init failed: {e}"}
        # Keep generating status frames so the video feed endpoint has something to show
        while True:
            global_frame = generate_status_frame("MODEL ERROR", str(e)[:50])
            time.sleep(1)
        return
        
    video_source = 1  # Secondary webcam
    
    # Use DirectShow on Windows to prevent MSMF crash spam
    if os.name == 'nt':
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(video_source)
    
    # Minimize camera buffer to avoid stale frames (huge latency reducer)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[WARNING] Live Camera could not be opened. Generating fallback frames.")
        camera_status = {"connected": False, "message": "Camera not available"}
        while True:
            global_frame = generate_status_frame("NO CAMERA", "Connect a webcam and restart the server")
            time.sleep(1)
        return
    
    camera_status = {"connected": True, "message": "Live camera active"}
    print("[BACKGROUND] Camera opened successfully. Starting optimized inference loop.")
        
    empty_start_time = None
    ALERT_DELAY_SECONDS = 5
    failed_frames = 0
    
    # --- Performance: run YOLO inference every Nth frame, reuse cached results ---
    INFERENCE_INTERVAL = 3  # Run YOLO every 3rd frame
    frame_idx = 0
    cached_person_count = 0
    cached_people_detections = []
    cached_appliance_count = 0
    cached_appliance_detections = []
    cached_brightness = 0.0
    cached_motion = 0
    cached_light_on = False
    cached_fan_running = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            failed_frames += 1
            if failed_frames > 15:
                print("[ERROR] Camera stream completely lost. Switching to fallback.")
                camera_status = {"connected": False, "message": "Camera stream lost"}
                cap.release()
                while True:
                    global_frame = generate_status_frame("SIGNAL LOST", "Camera disconnected")
                    time.sleep(1)
                return
            time.sleep(0.1)
            continue
            
        failed_frames = 0
        frame_idx += 1
            
        # 1. Run HEAVY inference only every Nth frame
        if frame_idx % INFERENCE_INTERVAL == 0:
            result = detector.detect_frame(frame)
            cached_person_count = result[0]
            cached_people_detections = result[1]
            cached_appliance_count = result[2]
            cached_appliance_detections = result[3]
            cached_light_on, cached_fan_running, cached_brightness, cached_motion = \
                appliance_detector.analyze_environment(frame, cached_people_detections)
        
        # Use cached results for every frame (fast)
        person_count = cached_person_count
        people_detections = cached_people_detections
        appliance_count = cached_appliance_count
        
        # Apply privacy ghost mode (stick figures + eye blur) — lightweight
        annotated = apply_ghost_mode(frame, people_detections)
        
        # Add HUD overlay text
        cv2.putText(annotated, f"OCCUPANTS: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 229, 255), 2, cv2.LINE_AA)
        status_text = "APPLIANCES: ON" if (cached_light_on or cached_fan_running) else "APPLIANCES: OFF"
        cv2.putText(annotated, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 1, cv2.LINE_AA)
        
        global_frame = annotated
        
        appliances_active = cached_light_on or cached_fan_running
        appliance_status_str = "ON" if appliances_active else "OFF"
        
        # 2. Logic processing
        alert = False
        if person_count == 0 and appliances_active:
            if empty_start_time is None:
                empty_start_time = time.time()
                
            if (time.time() - empty_start_time) >= ALERT_DELAY_SECONDS:
                alert = True
        else:
            empty_start_time = None
            
        # 3. Update the Global State for Room 101
        ROOMS_STATE[0]["person_count"] = person_count
        ROOMS_STATE[0]["appliance_state"] = appliance_status_str
        ROOMS_STATE[0]["appliance_count"] = appliance_count
        ROOMS_STATE[0]["alert"] = alert
        
        # Simulate savings
        if alert:
            ROOMS_STATE[0]["energy_saved_kwh"] += 0.005
        
        # Log history snapshot every ~3 seconds (every 100th frame at 30ms interval)
        if not hasattr(vision_processing_loop, '_frame_counter'):
            vision_processing_loop._frame_counter = 0
        vision_processing_loop._frame_counter += 1
        
        if vision_processing_loop._frame_counter % 100 == 0:
            snapshot = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "room": ROOMS_STATE[0]["id"],
                "person_count": person_count,
                "appliance_state": appliance_status_str,
                "alert": alert,
                "energy_saved_kwh": round(ROOMS_STATE[0]["energy_saved_kwh"], 4),
                "brightness": round(float(cached_brightness), 1),
                "motion_level": int(cached_motion)
            }
            history_log.append(snapshot)
            if len(history_log) > MAX_HISTORY:
                history_log[:] = history_log[-MAX_HISTORY:]
            
        # ~30 FPS display rate (inference is throttled separately via INFERENCE_INTERVAL)
        time.sleep(0.03)


# --- API LIFESPAN THREAD SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs when application starts
    vision_thread = threading.Thread(target=vision_processing_loop, daemon=True)
    vision_thread.start()
    yield
    # Runs when application shutdowns
    print("Shutting down API...")

app = FastAPI(title="Watt-Watch Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINTS ---
@app.get("/")
def read_root():
    return {"status": "Watt-Watch API is live. Background CV processing is active."}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Browsers automatically request a favicon. This blank response stops the 404 terminal spam.
    return Response(content=b"", media_type="image/x-icon")

@app.get("/api/status")
def get_room_status():
    """
    Returns the real-time calculated global state of the rooms securely.
    """
    return {
        "rooms": ROOMS_STATE
    }

@app.get("/api/camera_status")
def get_camera_status():
    return camera_status

@app.get("/api/history")
def get_history():
    """
    Returns historical KPI data for the analytics dashboard.
    If no live data yet, returns mock seed data for demo purposes.
    """
    if len(history_log) < 5:
        # Provide seed data for demo
        seed_data = [
            {"timestamp": "2026-03-26 08:00:00", "room": "Room 101 (Live Cam)", "person_count": 12, "appliance_state": "ON", "alert": False, "energy_saved_kwh": 0.0, "brightness": 145.2, "motion_level": 3200},
            {"timestamp": "2026-03-26 09:15:00", "room": "Room 101 (Live Cam)", "person_count": 0, "appliance_state": "ON", "alert": True, "energy_saved_kwh": 2.1, "brightness": 142.8, "motion_level": 800},
            {"timestamp": "2026-03-26 10:30:00", "room": "Room 102 (Class)", "person_count": 34, "appliance_state": "ON", "alert": False, "energy_saved_kwh": 5.5, "brightness": 160.0, "motion_level": 5500},
            {"timestamp": "2026-03-26 11:45:00", "room": "Room 103 (Lab)", "person_count": 0, "appliance_state": "OFF", "alert": False, "energy_saved_kwh": 8.1, "brightness": 45.3, "motion_level": 120},
            {"timestamp": "2026-03-26 12:00:00", "room": "Room 101 (Live Cam)", "person_count": 5, "appliance_state": "ON", "alert": False, "energy_saved_kwh": 8.1, "brightness": 155.0, "motion_level": 4100},
            {"timestamp": "2026-03-26 13:15:00", "room": "Room 102 (Class)", "person_count": 0, "appliance_state": "ON", "alert": True, "energy_saved_kwh": 10.3, "brightness": 138.5, "motion_level": 950},
            {"timestamp": "2026-03-26 14:30:00", "room": "Room 103 (Lab)", "person_count": 8, "appliance_state": "ON", "alert": False, "energy_saved_kwh": 12.0, "brightness": 170.2, "motion_level": 6200},
            {"timestamp": "2026-03-26 15:00:00", "room": "Room 101 (Live Cam)", "person_count": 0, "appliance_state": "OFF", "alert": False, "energy_saved_kwh": 14.5, "brightness": 30.1, "motion_level": 50},
        ]
        return {"history": seed_data + history_log}
    return {"history": history_log}

@app.get("/api/history/csv")
def get_history_csv():
    """
    Returns historical data as a downloadable CSV file.
    """
    data = get_history()["history"]
    
    output = io.StringIO()
    if data:
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    csv_content = output.getvalue()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=wattwatch_history.csv"}
    )

@app.get("/api/video_feed")
def video_feed():
    def generate():
        while True:
            if global_frame is not None:
                ret, buffer = cv2.imencode('.jpg', global_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
