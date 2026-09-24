# Future Dependencies

This file documents the dependencies that will be required for Phase 3 and beyond. They are deliberately omitted from the Phase 2 `requirements.txt` to keep the foundation clean and lightweight.

## Phase 3 (Detection & Recognition)
- `ultralytics` - For YOLOv8-Face detection.
- `onnxruntime` - CPU-optimized inference engine for both YOLOv8 and ArcFace.
- `faiss-cpu` - Facebook AI Similarity Search for rapid embedding matching.
- `filterpy` - Required for Kalman filters used in ByteTrack.
- `lapx` - Required for linear assignment (ByteTrack).

## Phase 5 (Dashboard & API)
- `fastapi` - Web framework for the dashboard API.
- `uvicorn` - ASGI server for FastAPI.
- `jinja2` - HTML templating for the web dashboard (if using SSR).
