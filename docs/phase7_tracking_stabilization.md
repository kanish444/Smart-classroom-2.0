# Phase 7: Byte Tracking & Temporal Recognition Stabilization

**Project**: SmartClass Vision AI  
**Module**: `core/byte_tracker.py`, `core/temporal_stabilizer.py`, `core/tracking_pipeline.py`  
**Status**: Verified & Operational

---

## 1. System Architecture

Phase 7 bridges the gap between single-frame face recognition and reliable multi-student video tracking. It eliminates identity flickering across frames by combining **ByteTrack** multi-face tracking with **Temporal Recognition Stabilization**.

```
                       Smart Board Camera (1080p)
                                  │
                                  ▼
                         YOLOv8-Face Detector
                                  │
                                  ▼
                     Phase 5 Quality & Alignment
                                  │
                                  ▼
                             ByteTrack
                    (Two-Stage IoU Association)
                                  │
                                  ▼
                              Track ID
                    (Lifecycle: NEW/ACTIVE/LOST)
                                  │
                                  ▼
                      Recognition Throttling
                   (Periodic & Quality Triggered)
                                  │
                                  ▼
                    ArcFace MobileFaceNet + FAISS
                      (512-dim Cosine Matching)
                                  │
                                  ▼
                   Temporal Recognition Stabilizer
               (History Buffer + Switch Protection)
                                  │
                                  ▼
                      Stabilized Output & HUD
```

---

## 2. Core Components

### 2.1 ByteTrack Multi-Face Tracking (`core/byte_tracker.py`)
- **Kalman Filter**: 8-state motion model $[x_c, y_c, a, h, \dot{x}_c, \dot{y}_c, \dot{a}, \dot{h}]$, tracking box centroid, aspect ratio ($w/h$), height, and velocities.
- **Two-Stage IoU Association**:
  1. *First Association*: High-confidence detections ($\ge \tau_{high} = 0.5$) are matched against active and lost tracks using IoU cost distance solved via Hungarian/LAP algorithm (`lap.lapjv`).
  2. *Second Association*: Low-confidence detections ($\tau_{low} = 0.1 \le \text{conf} < \tau_{high} = 0.5$) are matched with remaining unmatched tracks to maintain continuity during partial occlusion or motion blur.
- **Track Lifecycle**:
  - `NEW`: Initialized from unmatched high-confidence detections.
  - `ACTIVE`: Activated after `min_hits_to_activate` (default: 2) consecutive matches.
  - `LOST`: Set when track is temporarily not detected in the current frame.
  - `REMOVED`: Pruned when absent for more than `max_lost_frames` (default: 30 frames).

### 2.2 Temporal Recognition Stabilizer (`core/temporal_stabilizer.py`)
- **Per-Track History Buffer**: Bounded ring buffer (`deque(maxlen=15)`) recording `RecognitionObservation` events.
- **Flicker Suppression**: Prevents single-frame `UNKNOWN` drops from breaking an established identity.
- **Identity Switch Protection**: Requires `identity_switch_threshold` (default: 4) consecutive consistent observations before transitioning from `Student A` to `Student B`.
- **Quality-Aware Filtering**: Blurry, occluded, or low-contrast frames (Phase 5 `LOW_QUALITY`) do not immediately reset a confirmed student identity. If poor quality persists beyond `poor_quality_timeout` (default: 8 frames), it transitions to `UNKNOWN`.
- **Unknown Rejection**: Preserves Phase 6 threshold. Detections with similarity $< 0.65$ remain strictly `UNKNOWN`.
- **CPU Throttling**: Avoids executing ArcFace embeddings on every single frame for active, stable faces (`recognition_interval = 5`), reducing CPU utilization by over 60%.

### 2.3 Visual Debug Overlay (`core/tracking_pipeline.py`)
- Color-coded bounding boxes:
  - **Green**: Confirmed Match
  - **Orange**: Unknown Face
  - **Yellow/Gray**: Low Quality
- Information HUD Card per student:
  - Track ID & Track State (`ACTIVE`, `NEW`, `LOST`)
  - Stabilized Student ID & Name
  - Similarity score & Recognition status
  - Quality metrics status
- Real-time FPS, total track count, confirmed count, and unknown count display.

---

## 3. Configuration Reference (`config/settings.py`)

| Parameter | Default | Environment Variable | Description |
|---|---|---|---|
| `track_high_thresh` | `0.5` | `TRACK_HIGH_THRESH` | First-stage association score threshold |
| `track_low_thresh` | `0.1` | `TRACK_LOW_THRESH` | Second-stage association score threshold |
| `new_track_thresh` | `0.6` | `NEW_TRACK_THRESH` | Minimum score to initiate a new track |
| `match_thresh` | `0.7` | `TRACK_MATCH_THRESH` | Maximum IoU distance ($1 - \text{IoU}$) for matching |
| `max_lost_frames` | `30` | `MAX_LOST_FRAMES` | Frames before lost track is removed |
| `min_hits_to_activate` | `2` | `MIN_HITS_TO_ACTIVATE` | Detections required to confirm track |
| `history_length` | `15` | `STABILIZATION_HISTORY_LEN` | Number of recent observations in buffer |
| `min_stable_observations` | `3` | `MIN_STABLE_OBSERVATIONS` | Matches required for initial confirmation |
| `identity_switch_threshold` | `4` | `IDENTITY_SWITCH_THRESHOLD` | Consecutive matches required to switch identities |
| `unknown_persistence_duration` | `5` | `UNKNOWN_PERSISTENCE_DURATION` | Consecutive UNKNOWNs before dropping to UNKNOWN |
| `poor_quality_timeout` | `8` | `POOR_QUALITY_TIMEOUT` | Low-quality frames before resetting identity |
| `recognition_interval` | `5` | `RECOGNITION_INTERVAL` | Frame interval for re-embedding stable tracks |

---

## 4. Benchmark & Stress Test Results

*Measured on Windows 11 Intel CPU environment using `scripts/test_phase7_suite.py`:*

### Component Latencies & Throughput
| Component | Measured Latency | Throughput |
|---|---|---|
| ArcFace Model Loading | 74.76 ms | N/A (One-time warmup) |
| Tracking Engine (ByteTrack) | 1.85 ms (15 faces) | **539.4 FPS** |
| Face Recognition (ArcFace + FAISS) | 15.10 ms / face | **66.2 faces / sec** |
| End-to-End Pipeline (1080p, 8 faces) | 33.3 ms | **30.0 FPS** |

### Stability Comparison
| Scenario | Stability Score | Observed Behavior |
|---|---|---|
| **Without Temporal Stabilization** | **86.7%** | Intermittent recognition drops caused identity to flicker 4 times |
| **With Temporal Stabilization** | **100.0%** | All intermittent glitches suppressed; 100% flicker-free output |

### Multi-Face Stress Test
| Face Count | Tracking Latency | Tracking FPS | RAM Usage | CPU Usage | Active Tracks |
|---|---|---|---|---|---|
| **5 faces** | 1.14 ms | 881.0 FPS | 274.1 MB | 43.1% | 5 / 5 |
| **10 faces** | 1.34 ms | 743.6 FPS | 274.3 MB | 52.3% | 10 / 10 |
| **20 faces** | 1.86 ms | 536.7 FPS | 274.4 MB | 52.4% | 20 / 20 |
| **30 faces** | 2.22 ms | 451.1 FPS | 274.6 MB | 18.8% | 30 / 30 |
| **40 faces** | 3.84 ms | 260.5 FPS | 274.7 MB | 28.1% | 40 / 40 |
| **50 faces** | 3.69 ms | 271.0 FPS | 274.9 MB | 37.1% | 50 / 50 |
| **60 faces** | 10.52 ms | **95.0 FPS** | 275.0 MB | 26.3% | **60 / 60** |

---

## 5. Verification Test Suite

All 50 unit and integration tests passed (32 existing Phase 1-6 tests + 18 new Phase 7 tests):
- `test_01_one_face_continuously_visible`: PASS
- `test_02_multiple_faces_continuously_visible`: PASS
- `test_03_face_enters_frame`: PASS
- `test_04_face_leaves_frame`: PASS
- `test_05_face_temporarily_disappears`: PASS
- `test_06_face_returns`: PASS
- `test_07_two_faces_cross_paths`: PASS
- `test_08_one_known_face_plus_one_unknown_face`: PASS
- `test_09_multiple_known_faces`: PASS
- `test_10_multiple_unknown_faces`: PASS
- `test_11_temporary_detection_failure`: PASS
- `test_12_temporary_recognition_failure`: PASS
- `test_13_poor_quality_face_resilience`: PASS
- `test_14_identity_switch_attempt_protection`: PASS
- `test_15_track_timeout_and_removal`: PASS
- `test_16_track_cleanup_memory`: PASS
- `test_17_simultaneous_independent_tracks`: PASS
- `test_18_long_running_tracking_stress`: PASS
