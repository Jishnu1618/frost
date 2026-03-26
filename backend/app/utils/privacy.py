import cv2
import numpy as np

def apply_ghost_mode(frame, detections):
    '''
    Applies privacy-first anonymization to detected people.
    detections: list of tuples (x1, y1, x2, y2, confidence)
    '''
    anonymized_frame = frame.copy()
    
    for (x1, y1, x2, y2, conf) in detections:
        # Extract the region of interest
        roi = anonymized_frame[y1:y2, x1:x2]
        
        # Apply intense Gaussian Blur to anonymize the person
        if roi.size != 0:
            blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
            anonymized_frame[y1:y2, x1:x2] = blurred_roi
            
        # Draw bounding box (optional, just to show detection happened)
        cv2.rectangle(anonymized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
    return anonymized_frame
