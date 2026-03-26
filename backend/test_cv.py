import cv2
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.cv_model.detector import OccupancyDetector, ApplianceDetector
from app.utils.privacy import apply_ghost_mode

def main():
    print("Initializing YOLO-World model for people detection...")
    detector = OccupancyDetector()
    print("Initializing Physics Engine (Brightness & Motion) for Fan/Light detection...")
    appliance_detector = ApplianceDetector()
    
    # 0 is usually your laptop's built-in webcam.
    video_source = 0 
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}.")
        return

    print("Starting video stream. Press 'q' to quit.")
    
    # --- TIME SPAN ALERT LOGIC ---
    empty_start_time = None
    ALERT_DELAY_SECONDS = 5  # Time to wait before sending a message
    alert_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Detect People
        (person_count, people_detections, _, _) = detector.detect_frame(frame)
        
        # 2. Analyze Environment (Brightness & Running Fans)
        # Using pure OpenCV for states since YOLO struggles with "Running" vs "Static" and "Lit" vs "Unlit"
        light_is_on, fan_is_running, brightness, motion = appliance_detector.analyze_environment(frame, people_detections)
        
        # 3. Apply Privacy "Ghost Mode"
        anonymized_frame = apply_ghost_mode(frame, people_detections)
        
        # --- UI DISPLAY ---
        cv2.putText(anonymized_frame, f"Occupants: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(anonymized_frame, f"Light Level: {brightness:.1f} ({'ON' if light_is_on else 'OFF'})", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if light_is_on else (100, 100, 100), 2)
        cv2.putText(anonymized_frame, f"Motion Level: {motion} ({'RUNNING' if fan_is_running else 'STOPPED'})", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0) if fan_is_running else (100, 100, 100), 2)
        
        # --- TIMER & STATE LOGIC ---
        appliances_active = light_is_on or fan_is_running
        status_y = 140
        
        # Condition: Empty room BUT appliances are running/on
        if person_count == 0 and appliances_active:
            if empty_start_time is None:
                empty_start_time = time.time()  # Start the timer!
                alert_triggered = False
            
            elapsed = time.time() - empty_start_time
            remaining = max(0, ALERT_DELAY_SECONDS - elapsed)
            
            if remaining > 0:
                cv2.putText(anonymized_frame, f"WARNING: Empty but active! Alert in {remaining:.1f}s", (10, status_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(anonymized_frame, "ALERT TRIGGERED: MOCK SMS SENT!", (10, status_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                if not alert_triggered:
                    print("\n[MOCK API] -> Sending SMS Alert to Facility Manager: 'Room Empty. Devices left ON.'")
                    alert_triggered = True
        else:
            # Reset timer if people re-enter or appliances are turned off
            empty_start_time = None
            alert_triggered = False
            if person_count > 0:
                cv2.putText(anonymized_frame, "STATUS: OCCUPIED", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(anonymized_frame, "STATUS: SECURE (All Off)", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show the video feed
        cv2.imshow("Watt-Watch Demo", anonymized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
