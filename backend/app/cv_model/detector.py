import cv2
from ultralytics import YOLO

class OccupancyDetector:
    def __init__(self, model_path='yolov8s-world.pt'):
        # We upgraded to YOLO-World! 
        # This allows us to define custom equipment as text strings without doing manual training.
        print(f"Loading YOLO-World model from {model_path}...")
        self.model = YOLO(model_path)
        
        # Define the exact text phrases we want the AI to look for.
        self.custom_classes = [
            "person", 
            "ceiling fan", 
            "light", 
            "switch board", 
            "monitor", 
            "laptop"
        ]
        
        # Set the classes in the model (Zero-Shot Object Detection)
        self.model.set_classes(self.custom_classes)
        print("Model configured to detect:", self.custom_classes)
    
    def detect_frame(self, frame):
        # Run inference using the custom text classes
        results = self.model(frame, verbose=False)
        
        person_count = 0
        appliance_count = 0
        people_detections = []
        appliance_detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                
                # Get the string label for this detected class
                label = self.custom_classes[cls_id]
                
                # YOLO-World can return low confidence for text prompts, tune this if needed
                if conf < 0.05:
                    continue
                    
                if label == "person":
                    person_count += 1
                    people_detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
                else:
                    appliance_count += 1
                    appliance_detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), label))
                    
        return person_count, people_detections, appliance_count, appliance_detections

class ApplianceDetector:
    def __init__(self, history_frames=50):
        # Background subtractor to detect continuous motion (like a spinning fan blade)
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=history_frames, varThreshold=50, detectShadows=False)

    def analyze_environment(self, frame, people_detections):
        """
        Analyzes the frame for overall brightness and non-human motion.
        people_detections: used to ignore motion caused by people walking.
        """
        # 1. Brightness Check (Is the environment lit?)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2].mean()
        light_is_on = brightness > 90  # Adjust threshold based on your room brightness
        
        # 2. Motion Check (Is a fan running?)
        # Start by assuming motion in the frame could be anything
        fgMask = self.backSub.apply(frame)
        
        # To ensure we don't think a walking person is a ceiling fan, 
        # we black out the areas where people are detected in the motion mask
        for (x1, y1, x2, y2, conf) in people_detections:
            cv2.rectangle(fgMask, (x1, y1), (x2, y2), 0, -1)
            
        # Count remaining moving pixels (which would be fans, TVs, etc.)
        motion_level = cv2.countNonZero(fgMask)
        fan_is_running = motion_level > 1000  # Adjust threshold based on camera distance
        
        return light_is_on, fan_is_running, brightness, motion_level
