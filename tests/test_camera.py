import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.smartboard_camera import SmartBoardCamera
from camera.base_camera import BaseCamera

def test_camera_inheritance():
    cam = SmartBoardCamera(camera_index=0)
    assert isinstance(cam, BaseCamera)
    # We do not test actual frame grabbing in unit tests to prevent locking the device
    # or failing on CI/CD pipelines without cameras attached.
    cam.release()
