# SmartClass Vision AI

## Project Purpose
SmartClass Vision AI is a high-accuracy, single-camera multi-student face identification system designed to operate locally in a classroom setting using a Smart Board camera.

## Current Phase: Phase 8 (Completed)
**Phase 8 represents the Attendance Engine & Session Management Layer.**
- Decoupled identity architecture: Camera Detection ID != Track ID != Student ID != Attendance Record ID != Session ID
- Full session lifecycle state machine (SCHEDULED -> ACTIVE -> PAUSED -> ENDED / CANCELLED) with crash recovery
- Normalized SQLite persistence with foreign keys, cascades, and UNIQUE(session_id, student_id)
- Business rules: first_seen preservation, last_seen continuous update, late arrival policies, pre/post session windows
- In-memory sub-second event deduplication buffer (800+ frames/sec)
- Roster reconciliation reporting (PRESENT, LATE, NOT_SEEN)
- 92/92 automated unit tests passing across Phases 1-8

## System Requirements
- OS: Windows 10/11
- CPU: Intel i5 (or equivalent multi-core processor)
- RAM: 8GB+
- Python: >= 3.9
- Camera: Smart Board Camera (UVC compatible)

## Installation Instructions

1. Clone or copy the project folder.
2. Create the virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Copy `.env.example` to `.env` and adjust the settings.

## How to Run

**1. Run All Automated Unit Tests (Phases 1-8: 92 tests)**
```bash
pytest
```

**2. Run Phase 8 Attendance Benchmark & Performance Suite**
```bash
python scripts/test_phase8_suite.py
```

**3. Run Phase 7 Benchmark & Multi-Face Stress Test**
```bash
python scripts/test_phase7_suite.py
```

**3. Run Live Multi-Face Tracking HUD**
```bash
python scripts/test_tracking_live.py
```

## Current Functionality
- Full module directory structure
- Camera hardware abstraction & DirectShow fallback
- Centralized configuration using `pydantic` and `.env`
- Professional asynchronous logging using `loguru`
- Phase 4 YOLOv8 multi-face detection
- Phase 5 face quality assessment (size, blur, illumination), 5-point alignment, and normalization
- Phase 6 ArcFace (MobileFaceNet) 512-dim embedding generation & L2 normalization
- Phase 6 FAISS `IndexIDMap2` vector search with exact Cosine Similarity
- Phase 6 SQLite storage for student records and embedding BLOBs
- Phase 6 Unknown rejection and multi-face recognition

## What is NOT implemented yet (Deferred to Phase 7+)
- Face tracking & ID persistence across frames (ByteTrack)
- Temporal voting & smoothing across video frames
- Final production attendance database & reporting
- Real student biometric enrollments (uses sample/test identities only)
- Web dashboard

## Project Roadmap
- **Phase 1:** Architecture & Design (Done)
- **Phase 2:** Environment & Foundation (Done)
- **Phase 3:** Camera Pipeline (Done)
- **Phase 4:** Multi-Face Detection (Done)
- **Phase 5:** Quality Gating, Alignment & Normalization (Done)
- **Phase 6:** Face Embedding & Identity Matching Engine (Current - Complete)
- **Phase 7:** Tracking & Temporal Stabilization (Next)
- **Phase 8:** Attendance Engine & Dashboard

## Camera Architecture
The system uses a `BaseCamera` interface to abstract the physical camera hardware. The default implementation uses OpenCV to access the built-in Smart Board camera, with fallback logic for Windows DirectShow. 

## Privacy Note
All data processing happens locally on the host machine. No student faces or embeddings are sent to any cloud API. Real student data must NOT be placed in the `sample_data/` folder during the development phases.
