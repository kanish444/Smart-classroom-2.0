import sys
import os
import cv2

# Add parent directory to path to import local modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings

def run_diagnostic():
    print("=" * 40)
    print("CAMERA DIAGNOSTIC TOOL")
    print("=" * 40)
    
    settings = get_settings()
    index = settings.camera.index
    
    print(f"Attempting to connect to camera at index: {index}")
    
    # Try standard open
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print("Standard backend failed. Trying DirectShow (Windows)...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
    if not cap.isOpened():
        print("\nERROR: Camera is not exposed as an accessible device.")
        print(f"Failed to open camera index {index}.")
        print("Please check USB connections and Windows device manager.")
        sys.exit(1)
        
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("\nSTATUS: SUCCESS")
    print(f"Resolution: {width}x{height}")
    print(f"Reported FPS: {fps}")
    
    print("\nAttempting to read a single test frame...")
    ret, frame = cap.read()
    
    if ret:
        print(f"Frame read successfully! Shape: {frame.shape}")
        print("Note: Face recognition is NOT performed during this diagnostic.")
    else:
        print("ERROR: Opened camera but failed to grab a frame.")
        
    cap.release()
    print("=" * 40)

if __name__ == "__main__":
    run_diagnostic()
