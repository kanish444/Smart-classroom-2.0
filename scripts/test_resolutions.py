import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_settings

def test_resolutions():
    print("=" * 40)
    print("CAMERA RESOLUTION TESTER")
    print("=" * 40)
    
    settings = get_settings()
    index = settings.camera.index
    
    resolutions_to_test = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160)
    ]
    
    print(f"Testing camera index: {index}\n")
    
    for w, h in resolutions_to_test:
        print(f"Testing requested resolution: {w}x{h}")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("  Failed to open camera.\n")
            continue
            
        # Attempt to set
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        # Read back actual
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Test frame
        ret, frame = cap.read()
        if ret:
            print(f"  -> SUCCESS! Actual resolution provided: {actual_w}x{actual_h}")
            print(f"  -> Frame dimensions matched: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("  -> FAILED to read frame at this setting.")
            
        cap.release()
        print("-" * 40)

if __name__ == "__main__":
    test_resolutions()
