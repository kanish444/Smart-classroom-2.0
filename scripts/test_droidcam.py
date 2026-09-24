import os
import sys
import time
import argparse
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from camera.droidcam_camera import DroidCamCamera
from config.settings import get_settings


def test_droidcam(host: str, port: int, duration_sec: float = 3.0) -> Dict[str, Any]:
    """
    Performs comprehensive DroidCam connection validation:
    1. Base URL probe
    2. /video endpoint capture
    3. Frame dimension verification
    4. OpenCV decoding
    5. Continuous short stability stream
    """
    print("=" * 60)
    print("DROIDCAM CONNECTIVITY VERIFICATION TEST")
    print("=" * 60)
    print(f"Target Host: {host}")
    print(f"Target Port: {port}")
    print(f"Base Remote URL: http://{host}:{port}")
    print(f"Video Stream URL: http://{host}:{port}/video")
    print("-" * 60)

    # 1. Probe Port
    print("[1/5] Probing TCP connection to DroidCam server...")
    reachable, msg = DroidCamCamera.probe_endpoint(host, port, timeout=2.0)
    if not reachable:
        print(f"FAILED: Could not reach {host}:{port}")
        print(f"Reason: {msg}")
        print("\nDIAGNOSTIC GUIDANCE:")
        print("1. Ensure your phone and this laptop are connected to the SAME Wi-Fi network.")
        print("2. Ensure the DroidCam app is open and running on your phone.")
        print(f"3. Verify that the phone's Wi-Fi IP in DroidCam is indeed {host}.")
        print("4. Verify that DroidCam is using port 4747.")
        print("5. Check if Windows Firewall is blocking incoming/outgoing connections on port 4747.")
        return {
            "success": False,
            "stage": "tcp_probe",
            "message": f"DroidCam video stream unavailable at http://{host}:{port}/video. ({msg})"
        }
    print("SUCCESS: TCP connection established to DroidCam server.")

    # 2. Instantiate and Connect
    print(f"\n[2/5] Opening video stream at http://{host}:{port}/video via OpenCV...")
    cam = DroidCamCamera(host=host, port=port, video_path="/video")
    if not cam.is_connected:
        print("FAILED: DroidCam video stream unavailable.")
        print(cam.last_error_message)
        return {
            "success": False,
            "stage": "opencv_open",
            "message": cam.last_error_message
        }

    # 3. Read first frame
    print("\n[3/5] Verifying frame decoding and geometry...")
    success, frame = cam.get_frame()
    if not success or frame is None:
        print("FAILED: Connected to video stream, but failed to decode a valid video frame.")
        cam.release()
        return {
            "success": False,
            "stage": "frame_decode",
            "message": "Connected but frame decoding failed."
        }

    h, w, c = frame.shape
    print(f"SUCCESS: Frame decoded! Dimensions: {w}x{h} ({c} channels)")

    # 4. Stream Stability Test
    print(f"\n[4/5] Testing stream stability for {duration_sec:.1f} seconds...")
    t_start = time.time()
    frames_read = 0
    dropped = 0

    while time.time() - t_start < duration_sec:
        s, f = cam.get_frame()
        if s and f is not None:
            frames_read += 1
        else:
            dropped += 1
        time.sleep(0.02)

    actual_fps = cam.get_true_fps()
    print(f"SUCCESS: Stream active. Read {frames_read} frames (dropped: {dropped}). Measured FPS: {actual_fps:.1f}")

    # 5. Clean Release
    print("\n[5/5] Releasing camera stream cleanly...")
    cam.release()
    print("SUCCESS: Stream released.")

    print("=" * 60)
    print("OVERALL RESULT: DROIDCAM CONNECTED & FUNCTIONAL")
    print("=" * 60)

    return {
        "success": True,
        "width": w,
        "height": h,
        "measured_fps": actual_fps,
        "frames_read": frames_read,
        "video_url": cam.video_url
    }


if __name__ == "__main__":
    settings = get_settings().camera.droidcam
    default_host = getattr(settings, "host", "10.140.159.218")
    default_port = getattr(settings, "port", 4747)

    parser = argparse.ArgumentParser(description="Test DroidCam stream connectivity.")
    parser.add_argument("--host", default=default_host, help="DroidCam phone IP address")
    parser.add_argument("--port", type=int, default=default_port, help="DroidCam port")
    parser.add_argument("--duration", type=float, default=3.0, help="Test stream duration in seconds")
    args = parser.parse_args()

    result = test_droidcam(args.host, args.port, args.duration)
    sys.exit(0 if result["success"] else 1)
