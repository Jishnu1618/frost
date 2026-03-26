from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading
import cv2
import time
import os
from app.cv_model.detector import OccupancyDetector, ApplianceDetector

# --- GLOBAL STATE ---
# We store the latest processed CV results in memory so the API can serve it instantly.
# We include one LIVE room hooked up to your camera, and two MOCK rooms for dashboard aesthetics.
global_frame = None

ROOMS_STATE = [
    {
        "id": "Room 101 (Live Cam)",
        "person_count": 0,
        "appliance_state": "OFF",
        "alert": False,
        "energy_saved_kwh": 0.0
    },
    {
        "id": "Room 102 (Class)",
        "person_count": 34,
        "appliance_state": "ON",
        "alert": False,
        "energy_saved_kwh": 14.5
    },
    {
        "id": "Room 103 (Lab)",
        "person_count": 0,
        "appliance_state": "OFF",
        "alert": False,
        "energy_saved_kwh": 8.1
    }
]

def vision_processing_loop():
    """
    This function runs continuously in the background parsing the camera.
    """
    print("[BACKGROUND] Starting Vision Processing Thread...")
    try:
        detector = OccupancyDetector()
        appliance_detector = ApplianceDetector()
    except Exception as e:
        print("[WARNING] CV Models failed to initialize:", e)
        return
        
    video_source = 0  # Default webcam
    
    # Use DirectShow on Windows to prevent MSMF crash spam
    if os.name == 'nt':
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("[WARNING] Live Camera could not be opened. Using default Mock data.")
        return
        
    empty_start_time = None
    ALERT_DELAY_SECONDS = 5
    failed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            failed_frames += 1
            if failed_frames > 15:
                print("[ERROR] Camera stream completely lost. Shutting down Vision processing.")
                cap.release()
                return
            time.sleep(1)
            continue
            
        failed_frames = 0
            
        # 1. Run Detection
        global global_frame
        try:
            person_count, people_detections, _, annotated_frame = detector.detect_frame(frame)
        except ValueError:
            person_count, people_detections = detector.detect_frame(frame)[:2]
            annotated_frame = frame
            
        light_is_on, fan_is_running, brightness, motion = appliance_detector.analyze_environment(frame, people_detections)
        
        global_frame = annotated_frame.copy()
        
        appliances_active = light_is_on or fan_is_running
        appliance_status_str = "ON" if appliances_active else "OFF"
        
        # 2. Logic processing
        alert = False
        if person_count == 0 and appliances_active:
            if empty_start_time is None:
                empty_start_time = time.time()
                
            if (time.time() - empty_start_time) >= ALERT_DELAY_SECONDS:
                alert = True
                # In a real app we would fire an HTTP request or MQTT message to a Smart Plug here:
                # requests.post("http://smart-plug-ip/turn-off")
        else:
            empty_start_time = None
            
        # 3. Update the Global State for Room 101
        ROOMS_STATE[0]["person_count"] = person_count
        ROOMS_STATE[0]["appliance_state"] = appliance_status_str
        ROOMS_STATE[0]["alert"] = alert
        
        # Simulate savings: If the plug triggered off, we save energy!
        if alert:
            ROOMS_STATE[0]["energy_saved_kwh"] += 0.005 # Incremental savings for demo
            
        # Limit frame processing rate to save CPU power since this runs constantly
        time.sleep(0.3)


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
