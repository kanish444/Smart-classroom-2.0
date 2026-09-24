# Phase 6: Face Embedding & Identity Matching Engine

## Overview
Phase 6 implements the face recognition pipeline for SmartClass Vision AI, taking aligned and normalized face crops from Phase 5 and mapping them to authenticated student identities using deep face embeddings and vector similarity search.

---

## 1. ArcFace Model Information
- **Architecture:** MobileFaceNet (lightweight, optimized for edge/CPU inference)
- **Weights File:** `models/weights/w600k_mbf.onnx`
- **Training Dataset:** WebFace600K
- **Execution Engine:** ONNX Runtime (`CPUExecutionProvider`, 4 intra-op threads)
- **Input Signature:** `input.1` — Shape `[1, 3, 112, 112]`, Type `float32`
- **Output Signature:** `516` — Shape `[1, 512]`, Type `float32`

---

## 2. Embedding Dimensions & Preprocessing
- **Output Dimension:** 512 dimensions.
- **Input Dimensions:** `112 x 112` RGB spatial dimensions.
- **Phase 5 Normalization:** `(RGB.astype(float32) - 127.5) / 128.0`.
- **Channel Format:** Channel-first NCHW `(1, 3, 112, 112)`.

---

## 3. Embedding Normalization
- **Method:** Euclidean L2 Normalization:
  $$v_{\text{norm}} = \frac{v}{\max(\|v\|_2, 10^{-12})}$$
- **Verification:** Unit L2 norm $\|v_{\text{norm}}\|_2 = 1.0 \pm 10^{-4}$.
- **Mathematical Property:** For unit-normalized vectors, the inner product equals the exact cosine similarity:
  $$\cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u_{\text{norm}} \cdot v_{\text{norm}}$$

---

## 4. SQLite Database Schema
- **Database File:** `database/smartclass.sqlite`
- **Tables:**
  - `students`: `student_id` (PK), `student_name`, `department`, `section`, `status`, `created_at`, `updated_at`.
  - `embeddings`: `embedding_id` (PK AUTOINCREMENT), `student_id` (FK), `vector_blob` (BLOB of 512 float32s = 2048 bytes), `quality_score`, `sample_label`, `model_version`, `created_at`.
  - `model_metadata`: `id`, `model_name`, `embedding_dim`, `metric`, `threshold`, `updated_at`.

---

## 5. FAISS Vector Index Structure
- **Index Type:** `faiss.IndexIDMap2` wrapping `faiss.IndexFlatIP(512)`.
- **Metric:** Exact Inner Product (Cosine Similarity).
- **ID Synchronization:** FAISS vector IDs match SQLite `embedding_id` (int64) 1-to-1.
- **Persistence:** Saved to `database/faiss_index.bin` with metadata mapping `database/faiss_index.bin.meta.json`.
- **Consistency Validation:** `FaissVectorStore.validate_consistency()` verifies vector count against SQLite row count.

---

## 6. Similarity Metric & Unknown Rejection
- **Similarity Metric:** Cosine Similarity, strictly bounded in $[-1.0, 1.0]$.
  > [!NOTE]
  > Similarity is a metric distance score and must NOT be interpreted as a percentage probability of identity.
- **Configurable Threshold:** Default `0.65` (configurable via `RECOGNITION_THRESHOLD` in `.env`).
- **Decision Logic:**
  - $\max(\text{Similarity}) \ge \text{Threshold} \implies$ `MATCH` (identity assigned)
  - $\max(\text{Similarity}) < \text{Threshold} \implies$ `UNKNOWN` (no identity assigned)
  - Empty vector store or corrupted tensor $\implies$ `UNKNOWN` or `INVALID`

---

## 7. Multi-Sample Enrollment Architecture
- Supports enrolling multiple diverse samples per student identity (e.g. `frontal`, `slight_left`, `slight_right`, varied expressions).
- Every sample is validated through Phase 5 quality gating (area, sharpness, illumination) before embedding generation.
- No single photo dependency.

---

## 8. Structured Recognition Result
The `RecognitionResult` Pydantic model contains:
- `face_id`: Unique tracking/frame face identifier.
- `matched_student_id`: Assigned student ID if `MATCH`, otherwise `None`.
- `matched_student_name`: Assigned name if `MATCH`, otherwise `None`.
- `similarity`: Raw float cosine similarity score.
- `status`: `MATCH`, `UNKNOWN`, `INVALID`, or `ERROR`.
- `threshold`: Threshold used for the decision.
- `embedding_model`: `w600k_mbf.onnx`.
- `processing_time_ms`: End-to-end inference and search latency.
- `bbox`: Original bounding box coordinates.
- `quality_metrics`: Attached Phase 5 quality result.

---

## 9. Performance Measurements (Actual Hardware Benchmarks)
- **Model Loading:** `~77 ms`
- **Embedding Extraction:** `~18.6 ms` per face (CPU ONNX Runtime)
- **FAISS Vector Search:** `~5.8 ms` (for 15 vectors)
- **1-Face Recognition:** `~37.8 ms`
- **10-Face Batch:** `~302 ms` (`~30 ms` / face)
- **20-Face Batch:** `~512 ms` (`~25.6 ms` / face)
- **RAM Usage:** `~401.6 MB`
- **CPU Usage:** `~69.8%` during intensive 20-face multi-iteration stress

---

## 10. Known Limitations
1. **Resolution Degradation on Far Faces:**
   - In a 40ft classroom, faces in the FAR zone (< 3,000 px) are rejected by Phase 5 quality gating. ArcFace requires distinguishable landmarks for accurate embedding projection; far faces lack adequate inter-pupillary distance.
2. **Lighting Variations:**
   - Severe backlight or glare pushes faces outside the $[30, 230]$ brightness range, rejecting them prior to embedding.
3. **Absence of Temporal Smoothing (Deferred to Phase 7):**
   - Individual single-frame inferences can occasionally flutter near the threshold border. Temporal tracking and voting in Phase 7 will stabilize identity assignments across consecutive video frames.
