import cv2
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.camera_manager import CameraManager

def run_live_preview():
    print("Starting Live Preview Diagnostic...")
    print("Press 'q' to quit.")
    
    manager = CameraManager()
    
    window_name = "SmartClass Vision AI - Live Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        success, frame = manager.get_frame(max_retries=3)
        
        if not success:
            print("Failed to get frame. Exiting preview.")
            break
            
        status = manager.get_status()
        
        # Overlay diagnostic text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y0, dy = 30, 30
        
        texts = [
            "CAMERA DIAGNOSTIC",
            f"Source: Smart Board Camera",
            f"Resolution: {status.get('resolution', 'Unknown')}",
            f"Measured FPS: {status.get('measured_fps', 0.0)}",
            f"Status: {status.get('status', 'ERROR')}",
            f"Frames Read: {status.get('frames_read', 0)}"
        ]
        
        # Draw background rectangle for text readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 10 + len(texts)*dy + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        for i, text in enumerate(texts):
            y = y0 + i * dy
            cv2.putText(frame, text, (20, y), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    manager.release()
    cv2.destroyAllWindows()
    print("Live preview closed.")

if __name__ == "__main__":
    run_live_preview()
