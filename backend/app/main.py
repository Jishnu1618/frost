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
from pymongo import MongoClient
from app.cv_model.detector import OccupancyDetector, ApplianceDetector
from app.utils.privacy import apply_ghost_mode

# --- GLOBAL STATE ---
# We store the latest processed CV results in memory so the API can serve it instantly.
# We include one LIVE room hooked up to your camera, and two MOCK rooms for dashboard aesthetics.
global_frames = {0: None, 1: None, 2: None}
camera_status = {0: {"connected": False, "message": "Initializing..."},
                 1: {"connected": False, "message": "Initializing..."},
                 2: {"connected": False, "message": "Initializing..."}}
recording_states = {
    0: {"is_recording": False, "writer": None, "empty_timer": None},
    1: {"is_recording": False, "writer": None, "empty_timer": None},
    2: {"is_recording": False, "writer": None, "empty_timer": None}
}
history_log = []  # Stores timestamped snapshots of room state
MAX_HISTORY = 500  # Keep last 500 entries

# --- MONGODB INIT ---
MONGO_URI = "mongodb+srv://jishnuroy200316_db_user:Frost123@cluster0.unhvyy8.mongodb.net/?appName=Cluster0"
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["wattwatch"]
    history_collection = db["history_logs"]
    print("[INIT] Connected to MongoDB Atlas")
except Exception as e:
    print(f"[ERROR] Failed to connect to MongoDB: {e}")
    history_collection = None

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


def vision_processing_loop(room_index, video_source):
    """
    This function runs continuously in the background parsing the camera or video file.
    """
    global global_frames, camera_status, recording_states
    print(f"[BACKGROUND] Starting Vision Processing Thread for Room {room_index}...")
    
    try:
        detector = OccupancyDetector()
        appliance_detector = ApplianceDetector()
    except Exception as e:
        print("[WARNING] CV Models failed to initialize:", e)
        camera_status[room_index] = {"connected": False, "message": f"Model init failed: {e}"}
        while True:
            global_frames[room_index] = generate_status_frame("MODEL ERROR", str(e)[:50])
            time.sleep(1)
        return
        
    if isinstance(video_source, int) and os.name == 'nt':
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"[WARNING] Video source {video_source} could not be opened.")
        camera_status[room_index] = {"connected": False, "message": "Camera/Video not available"}
        while True:
            global_frames[room_index] = generate_status_frame("NO CAMERA", f"Source {video_source} failed")
            time.sleep(1)
        return
    
    camera_status[room_index] = {"connected": True, "message": "Video active"}
    print(f"[BACKGROUND] Source {video_source} opened successfully for Room {room_index}.")
        
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
            if isinstance(video_source, str):
                # Loop the video file back to the start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                failed_frames += 1
                if failed_frames > 15:
                    print(f"[ERROR] Camera stream {room_index} lost.")
                    camera_status[room_index] = {"connected": False, "message": "Camera stream lost"}
                    cap.release()
                    while True:
                        global_frames[room_index] = generate_status_frame("SIGNAL LOST", "Camera disconnected")
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
        
        global_frames[room_index] = annotated
        
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
            
        # 3. Update the Global State
        ROOMS_STATE[room_index]["person_count"] = person_count
        ROOMS_STATE[room_index]["appliance_state"] = appliance_status_str
        ROOMS_STATE[room_index]["appliance_count"] = appliance_count
        ROOMS_STATE[room_index]["alert"] = alert
        
        # Simulate savings
        if alert:
            ROOMS_STATE[room_index]["energy_saved_kwh"] += 0.005
            
        # Logging
        if not hasattr(vision_processing_loop, '_frame_counter'):
            vision_processing_loop._frame_counter = {}
        if room_index not in vision_processing_loop._frame_counter:
            vision_processing_loop._frame_counter[room_index] = 0
            
        vision_processing_loop._frame_counter[room_index] += 1
        
        if vision_processing_loop._frame_counter[room_index] % 100 == 0:
            snapshot = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "room": ROOMS_STATE[room_index]["id"],
                "person_count": person_count,
                "appliance_state": appliance_status_str,
                "alert": alert,
                "energy_saved_kwh": round(ROOMS_STATE[room_index]["energy_saved_kwh"], 4),
                "brightness": round(float(cached_brightness), 1),
                "motion_level": int(cached_motion)
            }
            if history_collection is not None:
                try:
                    history_collection.insert_one(snapshot.copy())
                except Exception as e:
                    pass

            history_log.append(snapshot)
            if len(history_log) > MAX_HISTORY:
                history_log[:] = history_log[-MAX_HISTORY:]
                
        # --- Recording Logic ---
        r_state = recording_states[room_index]
        if r_state["is_recording"]:
            if r_state["writer"] is None:
                h, w = frame.shape[:2]
                os.makedirs("data/records", exist_ok=True)
                filename = f"data/records/room{room_index}_{int(time.time())}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                r_state["writer"] = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
                print(f"[RECORDING] Started for room {room_index}: {filename}")
                
            r_state["writer"].write(annotated)
            # Add recording indicator UI to frame
            cv2.circle(annotated, (w-30, 30), 10, (0, 0, 255), -1)
            cv2.putText(annotated, "REC", (w-80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if person_count == 0:
                if r_state["empty_timer"] is None:
                    r_state["empty_timer"] = time.time()
                elif time.time() - r_state["empty_timer"] >= 5.0:
                    r_state["writer"].release()
                    r_state["writer"] = None
                    r_state["is_recording"] = False
                    r_state["empty_timer"] = None
                    print(f"[RECORDING] Stopped automatically for room {room_index} (empty for 5s)")
            else:
                r_state["empty_timer"] = None
            
        # ~30 FPS display rate
        time.sleep(0.03)


# --- API LIFESPAN THREAD SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs when application starts
    # Live Webcam
    threading.Thread(target=vision_processing_loop, args=(0, 0), daemon=True).start()
    # Mock offline videos
    threading.Thread(target=vision_processing_loop, args=(1, "data/room102/room102.mp4"), daemon=True).start()
    threading.Thread(target=vision_processing_loop, args=(2, "data/room103/room103.mp4"), daemon=True).start()
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
    if history_collection is not None:
        try:
            # Get the last 100 entries from DB
            cursor = history_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(100)
            db_history = list(cursor)[::-1]  # reverse to chronological order
            if len(db_history) > 0:
                return {"history": db_history}
        except Exception as e:
            print(f"[ERROR] Failed to fetch history from DB: {e}")

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

@app.get("/api/video_feed/{room_index}")
def video_feed(room_index: int):
    def generate():
        while True:
            frame = global_frames.get(room_index)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/record_video/{room_index}")
def api_record_video(room_index: int):
    if room_index in recording_states:
        recording_states[room_index]["is_recording"] = True
        # Reset the empty timer if it was set
        recording_states[room_index]["empty_timer"] = None
        return {"status": "success", "message": f"Recording started for room {room_index}"}
    return {"status": "error", "message": "Invalid room index"}, 400

@app.get("/api/recording_status/{room_index}")
def get_recording_status(room_index: int):
    if room_index in recording_states:
        return {"is_recording": recording_states[room_index]["is_recording"]}
    return {"error": "Invalid room index"}, 400
