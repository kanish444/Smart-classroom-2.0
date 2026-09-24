import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_directories_exist():
    required_dirs = ['app', 'camera', 'core', 'database', 'config', 'utils', 'logs', 'models/weights', 'sample_data']
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for d in required_dirs:
        full_path = os.path.join(base_path, d)
        assert os.path.exists(full_path), f"Directory {d} does not exist"

def test_python_version():
    assert sys.version_info >= (3, 9)
