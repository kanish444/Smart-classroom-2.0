# Phase 8: Attendance Engine & Session Management

**Project**: SmartClass Vision AI  
**Modules**: `attendance/session_manager.py`, `attendance/attendance_engine.py`, `database/db_manager.py`  
**Status**: Verified & Operational

---

## 1. System Architecture

Phase 8 introduces the business logic and database persistence layer for academic attendance. It operates downstream from Phase 7's multi-face tracking and temporal stabilization without modifying any upstream computer vision, alignment, or recognition components.

```
                        Phase 7 Tracked Faces
              (Stable Identities & Temporary Track IDs)
                                  │
                                  ▼
                          Attendance Event
            (session_id, student_id, track_id, timestamp)
                                  │
                                  ▼
                          Attendance Engine
           ┌──────────────────────┴──────────────────────┐
           ▼                                             ▼
  Session Window Check                          Deduplication Cache
 (SCHEDULED/ACTIVE/ENDED)                       (1.0s Sub-Second Buffer)
           │                                             │
           └──────────────────────┬──────────────────────┘
                                  ▼
                         Late Arrival Policy
                (first_seen <= start + 10m ? PRESENT : LATE)
                                  │
                                  ▼
                           SQLite Database
             ┌────────────────────┴────────────────────┐
             ▼                                         ▼
      sessions table                            attendance table
(Lifecycle: ACTIVE/ENDED)              UNIQUE(session_id, student_id)
                                        (first_seen preserved, last_seen updated)
```

---

## 2. Core Principles & Decoupled Identifiers

To guarantee database integrity and prevent tracking noise from affecting student records:

$$\text{Detection ID} \neq \text{Track ID} \neq \text{Student ID} \neq \text{Attendance Record ID} \neq \text{Session ID}$$

- **Track ID**: Temporary integer assigned by ByteTrack for motion association (e.g. `track_12`).
- **Student ID**: Persistent string identifier in the college database (e.g. `STU001`).
- **Session ID**: Unique identifier for an academic period (e.g. `AIDS_B_2026_09_23_09_15`).
- **Attendance Record ID**: Auto-incrementing primary key for an attendance entry.

---

## 3. Session Lifecycle Management (`SessionManager`)

Instructional sessions follow a strict, validated finite state machine:

```
          ┌─────────────┐
          │  SCHEDULED  │
          └──────┬──────┘
                 │ (start_session)
                 ▼
    ┌────────────┴────────────┐
    │                         │ (pause)
    ▼                         ▼
┌────────┐               ┌────────┐
│ ACTIVE │◄──────────────┤ PAUSED │
└───┬────┘   (resume)    └───┬────┘
    │                        │
    ├───────────┬────────────┤
    │ (end)     │ (cancel)   │ (cancel)
    ▼           ▼            ▼
┌───────┐   ┌───────────┐
│ ENDED │   │ CANCELLED │
└───────┘   └───────────┘
(Terminal)    (Terminal)
```

### Crash Recovery (`recover_sessions()`)
On application startup, the system scans for sessions left in non-terminal states (`ACTIVE`, `PAUSED`, `RECOVERING`):
- If the current time exceeds `planned_end_time`: automatically transitions the session to `ENDED`.
- If the session window is still open: sets status to `RECOVERING` to allow clean resumption.

---

## 4. Attendance Policy & Business Rules (`AttendanceEngine`)

### 4.1 First-Seen / Last-Seen Invariant
- **Rule**: Exactly **ONE** record per `(session_id, student_id)`.
- `first_seen`: Recorded when the student is first validly recognized. **Never overwritten** on subsequent recognitions.
- `last_seen`: Updated continuously as the student is re-observed in video frames.
- `PRESENT` status is permanent within the session: once marked `PRESENT`, a student is never demoted to `LATE`.

### 4.2 Late Arrival Policy
- Configurable threshold: `late_threshold_minutes` (default: 10.0 minutes).
- Let $T_{start}$ be the session start timestamp:
  - If $\text{first\_seen} \le T_{start} + \text{late\_threshold\_minutes}$: `PRESENT`
  - If $\text{first\_seen} > T_{start} + \text{late\_threshold\_minutes}$: `LATE`

### 4.3 Pre-Session and Post-Session Windows
- **Pre-Session**: If `allow_pre_session_marking` is true and a student is recognized within `pre_session_window_minutes` (default: 15 min) of planned start, they are marked `PRESENT`.
- **Post-Session**: Recognitions arriving after a session has `ENDED` or `CANCELLED` are rejected.

### 4.4 In-Memory Event Deduplication
- A sub-second deduplication buffer (`deduplication_buffer_seconds = 1.0s`) intercepts rapid consecutive frames (e.g. 30 FPS video) and avoids hundreds of redundant SQLite disk writes per minute, executing in **1.24 ms**.

### 4.5 Roster Reconciliation (`generate_session_report()`)
- Students enrolled in the class/section who were never validly observed are categorized as `NOT_SEEN`.

---

## 5. Normalized Database Schema

### `sessions` Table
```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    class_section TEXT NOT NULL,
    subject TEXT NOT NULL,
    planned_start_time TEXT NOT NULL,
    planned_end_time TEXT NOT NULL,
    actual_start_time TEXT,
    actual_end_time TEXT,
    status TEXT NOT NULL DEFAULT 'SCHEDULED',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(date);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
```

### `attendance` Table
```sql
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    status TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    first_track_id INTEGER,
    last_track_id INTEGER,
    initial_similarity REAL,
    latest_similarity REAL,
    marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    UNIQUE (session_id, student_id)
);
CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance(session_id);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_session_student ON attendance(session_id, student_id);
```

---

## 6. Performance & Benchmark Metrics

*Measured on Windows 11 Intel CPU environment using `scripts/test_phase8_suite.py`:*

| Operation | Measured Latency | Throughput |
|---|---|---|
| Session Creation | 12.092 ms | ~82 sessions/sec |
| Session Start | 9.386 ms | ~106 sessions/sec |
| Attendance Ingestion & SQLite Write (Mean) | 10.368 ms | **96.5 events/sec** |
| Attendance Ingestion & SQLite Write (P95) | 12.669 ms | N/A |
| Deduplicated In-Memory Filtering | 1.248 ms | **801.1 frames/sec** |
| Session End | 9.098 ms | ~110 sessions/sec |
| Session Report Generation (60 students) | 3.479 ms | **287 reports/sec** |
| Process RAM Usage | **418.6 MB** | Stable |
| Process CPU Usage | **53.6%** | During intensive write stress |

---

## 7. Verification Test Suite Summary

- **Total Tests Passing**: **92 / 92** (100% pass across all phases)
- **Phase 8 Specific Tests**: **42 / 42 passed in 6.93s**
- Zero regressions in Phases 1 through 7.
