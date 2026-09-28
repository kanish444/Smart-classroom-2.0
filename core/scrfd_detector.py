import os
import time
from typing import List, Any, Dict, Optional, Tuple
import cv2
import numpy as np
from loguru import logger
import onnxruntime as ort

from config.settings import get_settings
from core.detector import BaseDetector


def distance2bbox(points: np.ndarray, distance: np.ndarray, max_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Decodes distance predictions [left, top, right, bottom] from anchor points to [x1, y1, x2, y2].
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray, max_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Decodes 5-point facial keypoints [dx0, dy0, ... dx4, dy4] from anchor points.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFDDetector(BaseDetector):
    """
    High-Accuracy Multi-Face SCRFD Detector using ONNX Runtime:
    - Sample and Computation Redistribution for Efficient Face Detection (SCRFD).
    - Regresses tight face bounding boxes, confidence scores, and 5 facial keypoints.
    - Scales from single large close faces up to 60+ simultaneous students in a classroom.
    - Seamlessly integrates with existing FaceQuality, ArcFace, ByteTrack, FAISS, and Attendance.
    - Robust configuration-based model loading (never auto-downloads at application runtime).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        input_size: Optional[Tuple[int, int]] = None
    ):
        self.settings = get_settings().detection
        self.confidence_threshold = (
            conf_thresh if conf_thresh is not None else getattr(self.settings, "confidence_threshold", 0.40)
        )
        self.nms_threshold = (
            nms_thresh if nms_thresh is not None else getattr(self.settings, "nms_iou_threshold", 0.40)
        )
        self.padding_ratio = getattr(self.settings, "padding_ratio", 0.0)
        self.min_face_size = getattr(self.settings, "min_face_size", 16)
        self.enable_tiling = getattr(self.settings, "enable_tiling", False)
        self.tile_interval = getattr(self.settings, "tile_interval", 3)

        # 1. Resolve model file path
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate_path = model_path or getattr(self.settings, "model_path", None) or getattr(self.settings, "model_name", None)

        # Standard resolution candidates
        search_paths = []
        if candidate_path:
            if os.path.isabs(candidate_path):
                search_paths.append(candidate_path)
            else:
                search_paths.append(os.path.join(root_dir, candidate_path))
                search_paths.append(os.path.join(root_dir, "models", "face_detection", "scrfd", os.path.basename(candidate_path)))

        # Default fallback locations (Prioritizing SCRFD-10G-KPS)
        default_dir = os.path.join(root_dir, "models", "face_detection", "scrfd")
        search_paths.extend([
            os.path.join(default_dir, "scrfd_10g_kps.onnx"),
            os.path.join(default_dir, "scrfd_2.5g_kps.onnx"),
            os.path.join(default_dir, "scrfd_model.onnx"),
            os.path.join(root_dir, "models", "weights", "scrfd_10g_kps.onnx"),
            os.path.join(root_dir, "models", "weights", "scrfd_model.onnx"),
        ])

        resolved_path = None
        for p in search_paths:
            if os.path.exists(p) and os.path.isfile(p):
                resolved_path = p
                break

        if not resolved_path:
            msg = (
                f"[SCRFD_ERROR] SCRFD ONNX model file not found! "
                f"Searched paths: {search_paths}. "
                f"Please ensure 'scrfd_10g_kps.onnx' or 'scrfd_2.5g_kps.onnx' is placed in "
                f"'{default_dir}' or configure DETECTION_MODEL_PATH in environment."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        self.model_path = resolved_path
        logger.info(f"SCRFD_MODEL_LOADED: Initializing ONNX Runtime session from '{self.model_path}'...")

        # 2. Initialize ONNX Runtime Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = min(4, os.cpu_count() or 2)

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")

        try:
            self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
            logger.info(f"SCRFD Session created successfully with providers: {self.session.get_providers()}")
        except Exception as e:
            logger.error(f"[SCRFD_ERROR] Failed to create ONNX Runtime session for {self.model_path}: {e}")
            raise

        # 3. Model I/O Inspection
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        input_shape = inputs[0].shape  # e.g. [1, 3, 640, 640] or [1, 3, '?', '?']

        # Determine configured input resolution
        cfg_w = getattr(self.settings, "input_width", 640)
        cfg_h = getattr(self.settings, "input_height", 640)

        if input_size is not None:
            self.input_size = input_size
        elif len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            self.input_size = (input_shape[3], input_shape[2]) # (w, h)
        else:
            self.input_size = (cfg_w, cfg_h)

        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]

        # Landmark and output structure support (supports both named outputs 'score_8' and numerical '448')
        has_kps_named = any("kps" in name for name in self.output_names)
        has_kps_dim = any((len(o.shape) >= 2 and o.shape[-1] == 10) for o in outputs)
        self.has_kps = has_kps_named or has_kps_dim
        self.named_outputs = any("score_" in name for name in self.output_names)
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

        # Anchor centers cache keyed by (width, height)
        self._center_cache: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {}

        # 4. Model Warmup
        try:
            dummy_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
            self._infer_frame(dummy_img, conf_thresh=0.99)
            logger.info(f"SCRFD model warmup complete. Input size: {self.input_size}, Has 5-KPS: {self.has_kps}")
        except Exception as e:
            logger.warning(f"SCRFD warmup warning: {e}")

        # Metrics and state tracking
        self.latest_raw_detections: List[Dict[str, Any]] = []
        self.frame_count: int = 0
        self.last_detection_latency_ms: float = 0.0
        self.last_tile_latency_ms: float = 0.0

    def _get_anchor_centers(self, feat_w: int, feat_h: int, stride: int) -> np.ndarray:
        """Retrieves or precomputes anchor center coordinates for a feature map grid."""
        cache_key = (feat_w, feat_h)
        if cache_key not in self._center_cache:
            self._center_cache[cache_key] = {}
        if stride not in self._center_cache[cache_key]:
            grid_y, grid_x = np.meshgrid(np.arange(feat_h), np.arange(feat_w), indexing="ij")
            anchor_centers = np.stack([grid_x * stride, grid_y * stride], axis=-1).astype(np.float32)
            anchor_centers = np.repeat(anchor_centers[:, :, np.newaxis, :], self._num_anchors, axis=2).reshape((-1, 2))
            self._center_cache[cache_key][stride] = anchor_centers
        return self._center_cache[cache_key][stride]

    def get_zone(self, area: float) -> str:
        """Classifies face into NEAR, MIDDLE, or FAR distance zone."""
        if area > 15000:
            return "NEAR"
        elif area > 3000:
            return "MIDDLE"
        else:
            return "FAR"

    def apply_padding(self, x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Applies configured crop padding for recognition extraction."""
        if self.padding_ratio <= 0.0:
            return x1, y1, x2, y2
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * self.padding_ratio)
        pad_y = int(h * self.padding_ratio)
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(frame_w, x2 + pad_x)
        ny2 = min(frame_h, y2 + pad_y)
        return nx1, ny1, nx2, ny2

    def _infer_frame(
        self,
        image: np.ndarray,
        conf_thresh: float,
        target_size: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs SCRFD forward pass, anchor decoding, threshold filtering, and coordinate re-projection.
        """
        orig_h, orig_w = image.shape[:2]
        if orig_h == 0 or orig_w == 0:
            return []

        tw, th = target_size or self.input_size

        # 1. Aspect-Ratio Preserving Letterbox Resizing
        im_ratio = float(orig_h) / orig_w
        target_ratio = float(th) / tw

        if im_ratio > target_ratio:
            new_h = th
            new_w = max(1, int(new_h / im_ratio))
        else:
            new_w = tw
            new_h = max(1, int(new_w * im_ratio))

        # 0. Adaptive Low-Light Enhancement (adapted from reference project for dark classroom scenes)
        proc_img = image
        gray_sample = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray_sample)) < 65.0:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            l_chan = clahe.apply(l_chan)
            proc_img = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

        det_scale = float(new_h) / orig_h
        resized_img = cv2.resize(proc_img, (new_w, new_h))
        det_img = np.zeros((th, tw, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized_img

        # 2. Normalization: BGR -> RGB, (x - 127.5) / 128.0, NCHW layout
        blob = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        blob = (blob.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        # 3. ONNX Runtime Forward Pass
        outputs = self.session.run(None, {self.input_name: blob})
        out_dict = dict(zip(self.output_names, outputs))

        # Build stride map handling both named outputs ('score_8') and numerical outputs ('448')
        stride_data = {}
        if self.named_outputs:
            for stride in self._feat_stride_fpn:
                s_arr = out_dict.get(f"score_{stride}")
                b_arr = out_dict.get(f"bbox_{stride}")
                k_arr = out_dict.get(f"kps_{stride}")
                if s_arr is not None and b_arr is not None:
                    s_2d = s_arr[0] if s_arr.ndim == 3 else s_arr
                    b_2d = b_arr[0] if b_arr.ndim == 3 else b_arr
                    k_2d = (k_arr[0] if k_arr.ndim == 3 else k_arr) if k_arr is not None else None
                    stride_data[stride] = (s_2d, b_2d, k_2d)
        else:
            # Rank outputs by anchor count
            scores_outs = sorted(
                [o for o in outputs if o.shape[-1] == 1],
                key=lambda x: x.shape[0] if x.ndim == 2 else x.shape[1],
                reverse=True
            )
            bboxes_outs = sorted(
                [o for o in outputs if o.shape[-1] == 4],
                key=lambda x: x.shape[0] if x.ndim == 2 else x.shape[1],
                reverse=True
            )
            kps_outs = sorted(
                [o for o in outputs if o.shape[-1] == 10],
                key=lambda x: x.shape[0] if x.ndim == 2 else x.shape[1],
                reverse=True
            )
            for idx, stride in enumerate(self._feat_stride_fpn):
                s_arr = scores_outs[idx] if idx < len(scores_outs) else None
                b_arr = bboxes_outs[idx] if idx < len(bboxes_outs) else None
                k_arr = kps_outs[idx] if idx < len(kps_outs) else None
                if s_arr is not None and b_arr is not None:
                    s_2d = s_arr[0] if s_arr.ndim == 3 else s_arr
                    b_2d = b_arr[0] if b_arr.ndim == 3 else b_arr
                    k_2d = (k_arr[0] if k_arr.ndim == 3 else k_arr) if k_arr is not None else None
                    stride_data[stride] = (s_2d, b_2d, k_2d)

        scores_list = []
        bboxes_list = []
        kps_list = []

        # 4. Multi-Stride FPN Anchor Decoding
        for stride in self._feat_stride_fpn:
            if stride not in stride_data:
                continue

            score, bbox, kps = stride_data[stride]
            has_stride_kps = (kps is not None)

            feat_h = th // stride
            feat_w = tw // stride
            anchors = self._get_anchor_centers(feat_w, feat_h, stride)

            # Filter candidates above confidence threshold
            pos_inds = np.where(score.flatten() >= conf_thresh)[0]
            if len(pos_inds) > 0:
                pos_scores = score[pos_inds, 0]
                pos_bbox = bbox[pos_inds] * stride
                pos_anchors = anchors[pos_inds]

                det_bboxes = distance2bbox(pos_anchors, pos_bbox)
                scores_list.append(pos_scores)
                bboxes_list.append(det_bboxes)

                if has_stride_kps and kps is not None:
                    pos_kps = kps[pos_inds] * stride
                    det_kps = distance2kps(pos_anchors, pos_kps)
                    kps_list.append(det_kps)
                else:
                    kps_list.append(np.zeros((len(pos_inds), 10), dtype=np.float32))

        if len(scores_list) == 0:
            return []

        all_scores = np.concatenate(scores_list, axis=0)
        all_bboxes = np.concatenate(bboxes_list, axis=0)
        all_kps = np.concatenate(kps_list, axis=0)

        # 5. Map coordinates back to original frame
        all_bboxes[:, 0] = all_bboxes[:, 0] / det_scale
        all_bboxes[:, 1] = all_bboxes[:, 1] / det_scale
        all_bboxes[:, 2] = all_bboxes[:, 2] / det_scale
        all_bboxes[:, 3] = all_bboxes[:, 3] / det_scale

        all_kps[:, 0::2] = all_kps[:, 0::2] / det_scale
        all_kps[:, 1::2] = all_kps[:, 1::2] / det_scale

        # 6. Apply NMS
        order = all_scores.argsort()[::-1]
        keep = []
        x1 = all_bboxes[:, 0]
        y1 = all_bboxes[:, 1]
        x2 = all_bboxes[:, 2]
        y2 = all_bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w_inter = np.maximum(0.0, xx2 - xx1)
            h_inter = np.maximum(0.0, yy2 - yy1)
            inter = w_inter * h_inter
            ovr = inter / np.maximum(1e-6, areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]

        candidates = []
        for idx in keep:
            raw_x1, raw_y1, raw_x2, raw_y2 = all_bboxes[idx]
            conf = float(all_scores[idx])

            # Integer coordinate rounding and boundary clamping
            bx1 = max(0, min(int(round(raw_x1)), orig_w - 1))
            by1 = max(0, min(int(round(raw_y1)), orig_h - 1))
            bx2 = max(bx1 + 1, min(int(round(raw_x2)), orig_w))
            by2 = max(by1 + 1, min(int(round(raw_y2)), orig_h))

            bw = bx2 - bx1
            bh = by2 - by1

            if bw < self.min_face_size or bh < self.min_face_size:
                continue

            # Aspect-ratio validation (filter non-facial vertical/horizontal artifacts)
            aspect_ratio = bw / float(max(1, bh))
            if aspect_ratio < 0.35 or aspect_ratio > 2.2:
                continue

            tight_bbox = [bx1, by1, bx2, by2]
            crop_x1, crop_y1, crop_x2, crop_y2 = self.apply_padding(bx1, by1, bx2, by2, orig_w, orig_h)
            crop_bbox = [crop_x1, crop_y1, crop_x2, crop_y2]
            area = bw * bh
            zone = self.get_zone(area)

            # Extract 5 facial keypoints
            raw_pts = all_kps[idx]
            extracted_pts = []
            has_valid_pts = False
            if self.has_kps and raw_pts is not None and len(raw_pts) >= 10:
                valid_count = 0
                for p in range(5):
                    px = round(float(raw_pts[p * 2]), 1)
                    py = round(float(raw_pts[p * 2 + 1]), 1)
                    if (bx1 - 0.25 * bw) <= px <= (bx2 + 0.25 * bw) and (by1 - 0.25 * bh) <= py <= (by2 + 0.25 * bh):
                        valid_count += 1
                    extracted_pts.append([px, py])
                if valid_count >= 4 and len(extracted_pts) == 5:
                    has_valid_pts = True

            if not has_valid_pts:
                # Proportional facial landmark fallback
                extracted_pts = [
                    [round(bx1 + bw * 0.31, 1), round(by1 + bh * 0.38, 1)],
                    [round(bx1 + bw * 0.69, 1), round(by1 + bh * 0.38, 1)],
                    [round(bx1 + bw * 0.50, 1), round(by1 + bh * 0.58, 1)],
                    [round(bx1 + bw * 0.34, 1), round(by1 + bh * 0.78, 1)],
                    [round(bx1 + bw * 0.66, 1), round(by1 + bh * 0.78, 1)],
                ]

            candidates.append({
                "bbox": tight_bbox,
                "crop_bbox": crop_bbox,
                "confidence": conf,
                "face_area": area,
                "zone": zone,
                "class_id": 0,
                "keypoints": extracted_pts
            })

        return candidates

    def _get_overlapping_tiles(self, frame_w: int, frame_h: int) -> List[Tuple[int, int, int, int]]:
        """Generates 2 overlapping horizontal tiles for the distant student zone (y: 6% to 68%)."""
        y1 = max(0, int(0.06 * frame_h))
        y2 = min(frame_h, int(0.68 * frame_h))
        t1_x1 = 0
        t1_x2 = min(frame_w, int(0.60 * frame_w))
        t2_x1 = max(0, int(0.40 * frame_w))
        t2_x2 = frame_w
        return [(t1_x1, y1, t1_x2, y2), (t2_x1, y1, t2_x2, y2)]

    def _deduplicate_detections(
        self,
        candidate_detections: List[Dict[str, Any]],
        iou_thresh: float = 0.45,
        containment_thresh: float = 0.85
    ) -> List[Dict[str, Any]]:
        """Secondary deduplication across full-frame and tiled detections."""
        candidate_detections.sort(key=lambda d: d["confidence"], reverse=True)
        final_detections: List[Dict[str, Any]] = []

        for cand in candidate_detections:
            cx1, cy1, cx2, cy2 = cand["bbox"]
            c_area = cand["face_area"]
            is_duplicate = False

            for accepted in final_detections:
                ax1, ay1, ax2, ay2 = accepted["bbox"]
                a_area = accepted["face_area"]

                ix1 = max(cx1, ax1)
                iy1 = max(cy1, ay1)
                ix2 = min(cx2, ax2)
                iy2 = min(cy2, ay2)

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter_area = iw * ih

                if inter_area > 0:
                    union_area = c_area + a_area - inter_area
                    iou = inter_area / max(1e-6, union_area)
                    min_area = min(c_area, a_area)
                    iom = inter_area / max(1e-6, min_area)

                    if iou > iou_thresh or iom > containment_thresh:
                        is_duplicate = True
                        break

            if not is_duplicate:
                final_detections.append(cand)

        return final_detections

    def detect(self, frame: Any, run_tiles: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Executes multi-face detection using SCRFD:
        1. Full-frame inference with aspect-ratio preserving letterbox.
        2. Optional tiled detection for recovering small/distant background faces.
        3. Precise coordinate mapping back to original frame resolution.
        4. Secondary NMS deduplication.

        Args:
            frame: Input BGR image frame (numpy ndarray)
            run_tiles: True to force tiling, False to skip, None to use configured schedule.

        Returns:
            List of detected faces with tight bounding boxes, confidence, keypoints, and zone.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("[SCRFD_ERROR] Empty or invalid frame passed to detector.")
            return []

        t0 = time.perf_counter()
        self.frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        conf_thresh = self.confidence_threshold

        # 1. Full-Frame SCRFD Detection
        all_candidates = self._infer_frame(frame, conf_thresh=conf_thresh)

        # 2. Controlled Tiled Detection (Optional small-face recovery for classroom)
        should_tile = run_tiles if run_tiles is not None else (
            self.enable_tiling and (self.frame_count % self.tile_interval == 0)
        )

        t_tile_0 = time.perf_counter()
        if should_tile and frame_w >= 640 and frame_h >= 360:
            tiles = self._get_overlapping_tiles(frame_w, frame_h)
            for tx1, ty1, tx2, ty2 in tiles:
                tile_crop = frame[ty1:ty2, tx1:tx2]
                if tile_crop.size == 0:
                    continue

                tile_candidates = self._infer_frame(tile_crop, conf_thresh=conf_thresh)

                for tc in tile_candidates:
                    bx1, by1, bx2, by2 = tc["bbox"]
                    orig_x1 = max(0, min(frame_w - 1, tx1 + bx1))
                    orig_y1 = max(0, min(frame_h - 1, ty1 + by1))
                    orig_x2 = max(0, min(frame_w, tx1 + bx2))
                    orig_y2 = max(0, min(frame_h, ty1 + by2))

                    orig_w = orig_x2 - orig_x1
                    orig_h = orig_y2 - orig_y1
                    if orig_w <= 0 or orig_h <= 0:
                        continue

                    cbx1, cby1, cbx2, cby2 = tc["crop_bbox"]
                    orig_crop = [
                        max(0, min(frame_w - 1, tx1 + cbx1)),
                        max(0, min(frame_h - 1, ty1 + cby1)),
                        max(0, min(frame_w, tx1 + cbx2)),
                        max(0, min(frame_h, ty1 + cby2))
                    ]

                    mapped_kpts = [
                        [round(tx1 + pt[0], 1), round(ty1 + pt[1], 1)]
                        for pt in tc.get("keypoints", [])
                    ]

                    all_candidates.append({
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "crop_bbox": orig_crop,
                        "confidence": tc["confidence"],
                        "face_area": orig_w * orig_h,
                        "zone": self.get_zone(orig_w * orig_h),
                        "class_id": tc["class_id"],
                        "keypoints": mapped_kpts
                    })

        self.last_tile_latency_ms = (time.perf_counter() - t_tile_0) * 1000.0

        # 3. Deduplication & NMS
        final_detections = self._deduplicate_detections(all_candidates, iou_thresh=self.nms_threshold)
        self.last_detection_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latest_raw_detections = final_detections

        # 4. Structured Logging (Section 22)
        cnt_small = sum(1 for d in final_detections if d["face_area"] < 3000)
        cnt_lq = sum(1 for d in final_detections if d["face_area"] < 1600)
        logger.debug(
            f"SCRFD_FRAME_PROCESSED: faces={len(final_detections)}, "
            f"small_faces={cnt_small}, low_quality={cnt_lq}, "
            f"inference_ms={self.last_detection_latency_ms:.1f}"
        )

        return final_detections

    def detect_raw(self, frame: Any) -> List[Dict[str, Any]]:
        """Convenience method returning raw detections."""
        return self.detect(frame)

    def detect_faces(self, frame: Any) -> List[Dict[str, Any]]:
        """Alias for detect() for legacy enrollment callers."""
        return self.detect(frame)

    def __call__(self, frame: Any) -> List[Dict[str, Any]]:
        """Callable interface forwarding to detect()."""
        return self.detect(frame)

    def detect_results(self, frame: Any) -> List[Any]:
        """
        Returns FaceDetectionResult schema objects (Section 3 of requirements).
        """
        from core.schemas import FaceDetectionResult
        raw = self.detect(frame)
        return [
            FaceDetectionResult(
                bbox=d["bbox"],
                confidence=d["confidence"],
                keypoints=d.get("keypoints"),
                face_area=d.get("face_area", 0),
                crop_bbox=d.get("crop_bbox"),
                zone=d.get("zone", "FAR"),
                class_id=d.get("class_id", 0)
            )
            for d in raw
        ]

