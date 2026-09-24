import cv2

def list_cameras():
    print("Detecting accessible cameras using DirectShow...")
    available_cameras = []
    
    # Check first 10 indices
    for index in range(10):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            available_cameras.append({
                "index": index,
                "resolution": f"{width}x{height}",
                "fps": fps
            })
            cap.release()
            
    if not available_cameras:
        print("No cameras found.")
    else:
        for cam in available_cameras:
            print(f"Camera Index {cam['index']}: {cam['resolution']} @ {cam['fps']} FPS")

if __name__ == "__main__":
    list_cameras()
