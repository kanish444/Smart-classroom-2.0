import sys
from app import __version__
from config.settings import get_settings
from core.logger import setup_logger
from utils.validation import validate_environment

def main():
    # 1. Load configuration
    settings = get_settings()
    
    # 2. Initialize logging
    logger = setup_logger()
    
    # 3. Validate the environment
    is_valid = validate_environment()
    
    # 4. Display application version & status
    print("=" * 40)
    print("SMARTCLASS VISION AI")
    print("=" * 40)
    print(f"Version: {__version__}")
    print(f"Environment: {settings.environment.capitalize()}")
    print(f"Camera Source: Smart Board Camera (Index {settings.camera.index})")
    print("Recognition Engine: Not initialized")
    print("Database: Development mode")
    
    if is_valid:
        print("Status: Environment OK")
    else:
        print("Status: Environment Check Failed. Check logs for details.")
    print("=" * 40)

    # 5. Launch Smart Board Dashboard server
    if "--check-only" not in sys.argv:
        import uvicorn
        import signal
        from app.state import get_app_state

        def _handle_exit(sig, frame):
            print(f"\nShutdown signal ({sig}) received. Releasing camera hardware...")
            get_app_state().stop_camera_worker()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_exit)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handle_exit)

        port = 8008
        host = "0.0.0.0"
        for arg in sys.argv:
            if arg.startswith("--port="):
                port = int(arg.split("=")[1])
            elif arg.startswith("--host="):
                host = arg.split("=")[1]

        import socket
        local_ips = []
        try:
            hostname = socket.gethostname()
            for ip in socket.gethostbyname_ex(hostname)[2]:
                if not ip.startswith("127."):
                    local_ips.append(ip)
        except Exception:
            pass

        print(f"Launching Smart Board Dashboard server:")
        print(f"  * Localhost:    http://localhost:{port}")
        print(f"  * Loopback:     http://127.0.0.1:{port}")
        for ip in local_ips:
            print(f"  * Network URL:  http://{ip}:{port}")
        print(f"Binding to {host}:{port}...\n")
        try:
            uvicorn.run("app.main:app", host=host, port=port, reload=False)
        except (KeyboardInterrupt, SystemExit):
            print("\nShutting down server and releasing camera hardware...")
        finally:
            get_app_state().stop_camera_worker()
        sys.exit(0)

    # 6. Exit cleanly
    if not is_valid:
        sys.exit(1)
        
    sys.exit(0)

if __name__ == "__main__":
    main()

