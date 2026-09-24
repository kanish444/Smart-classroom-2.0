import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings

def test_settings_load():
    settings = get_settings()
    assert settings is not None
    assert hasattr(settings, 'environment')
    assert hasattr(settings, 'camera')
    assert settings.camera.index >= 0
    assert settings.camera.width > 0
    assert settings.camera.height > 0
