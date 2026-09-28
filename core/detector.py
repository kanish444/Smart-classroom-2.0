import os
import time
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, Tuple
import numpy as np
from loguru import logger
from config.settings import get_settings

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics package is missing. YOLOv8FaceDetector will not function.")


class BaseDetector(ABC):
    """
    Abstract base class for face detectors.
    """
    @abstractmethod
    def detect(self, frame: Any) -> List[Dict[str, Any]]:
        pass


class YOLOv8FaceDetector(BaseDetector):
    """
    YOLOv8-based Face Detector with Small-Face & Multi-Scale Recovery:
    - Uses ultralytics YOLOv8-Face targeting tight face bounding boxes and 5 facial landmarks.
    - Full-frame high-resolution inference (imgsz=960) detects small and distant classroom faces.
    - Controlled multi-scale overlapping tiling across upper/middle classroom zone recovers
      distant background students without excessive latency.
    - Strict secondary NMS and containment deduplication prevents duplicate boxes for the same student.
    - Bounding boxes are accurately mapped back to the original full-frame coordinate system.
    """

    def __init__(self, model_path: str = None):
        self.settings = get_settings().detection
        self.model_path = model_path or self.settings.model_name
        self.confidence_threshold = self.settings.confidence_threshold
        self.padding_ratio = self.settings.padding_ratio
        self.input_size = getattr(self.settings, "input_size", 960)
        self.enable_tiling = getattr(self.settings, "enable_tiling", True)
        self.tile_interval = getattr(self.settings, "tile_interval", 3)

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_dir = os.path.join(root_dir, "models", "weights")
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir, exist_ok=True)

        target_name = self.model_path
        if target_name in ["yolov8n.pt", "yolov8n", "coco"]:
            logger.warning(
                f"Requested detector model '{target_name}' is a COCO full-body detector. "
                "Redirecting to 'yolov8n-face.pt' for tight, accurate face bounding boxes."
            )
            target_name = "yolov8n-face.pt"

        final_model_path = os.path.join(weights_dir, target_name)
        weights_face_path = os.path.join(weights_dir, "yolov8n-face.pt")
        root_face_path = os.path.join(root_dir, "yolov8n-face.pt")
        root_model_path = os.path.join(root_dir, target_name)

        chosen_path = None
        if os.path.exists(final_model_path) and "face" in os.path.basename(final_model_path):
            chosen_path = final_model_path
        elif os.path.exists(weights_face_path):
            chosen_path = weights_face_path
        elif os.path.exists(root_face_path):
            chosen_path = root_face_path
        elif os.path.exists(root_model_path) and "face" in os.path.basename(root_model_path):
            chosen_path = root_model_path
        elif os.path.exists(final_model_path):
            chosen_path = final_model_path
        else:
            raise FileNotFoundError(
                f"YOLOv8 face detection model not found. Expected 'yolov8n-face.pt' in "
                f"'{weights_dir}' or '{root_dir}'. Do NOT substitute with generic person weights."
            )

        logger.info(f"Loading YOLOv8 Face model from {chosen_path}...")
        try:
            self.model = YOLO(chosen_path)
            # Force a dry run to warm up model in memory
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_img, verbose=False)
            logger.info("YOLOv8 Face Detector loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

        self.latest_raw_detections: List[Dict[str, Any]] = []
        self.frame_count: int = 0
        self.last_detection_latency_ms: float = 0.0
        self.last_tile_latency_ms: float = 0.0

    def get_zone(self, area: float, frame_width: int, frame_height: int) -> str:
        """
        Classifies the face into NEAR, MIDDLE, or FAR zone based on bounding box area.
        """
        if area > 15000:
            return "NEAR"
        elif area > 3000:
            return "MIDDLE"
        else:
            return "FAR"

    def apply_padding(self, x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int) -> tuple:
        """
        Applies a percentage padding to the bounding box strictly for recognition crop extraction.
        """
        w = x2 - x1
        h = y2 - y1

        ratio = getattr(self.settings, "padding_ratio", 0.0)
        if ratio <= 0.0:
            return x1, y1, x2, y2

        pad_x = int(w * ratio)
        pad_y = int(h * ratio)

        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(frame_w, x2 + pad_x)
        ny2 = min(frame_h, y2 + pad_y)

        return nx1, ny1, nx2, ny2

    def _infer_image(
        self,
        image: np.ndarray,
        conf_thresh: float,
        imgsz: int,
        iou_thresh: float = 0.45,
        max_det: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Runs YOLOv8 forward inference on a given image (full frame or tile crop)
        and extracts candidate face bounding boxes and 5 landmarks.
        """
        img_h, img_w = image.shape[:2]
        try:
            results = self.model(
                image,
                verbose=False,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=imgsz,
                max_det=max_det
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []

        if len(results) == 0:
            return []

        boxes = results[0].boxes
        keypoints_data = results[0].keypoints.data if results[0].keypoints is not None else None
        min_face_size = getattr(self.settings, "min_face_size", 20)

        candidates = []
        for i, box in enumerate(boxes):
            box_xyxy = box.xyxy[0]
            if hasattr(box_xyxy, "cpu"):
                xyxy = box_xyxy.cpu().numpy().tolist()
            else:
                xyxy = list(box_xyxy)

            box_conf = box.conf[0]
            conf = float(box_conf.cpu().numpy()) if hasattr(box_conf, "cpu") else float(box_conf)

            box_cls = box.cls[0]
            cls = int(box_cls.cpu().numpy()) if hasattr(box_cls, "cpu") else int(box_cls)

            # Enforce face class if model defines class names
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                cls_name = str(self.model.names.get(cls, "")).lower()
                all_names = [str(n).lower() for n in self.model.names.values()]
                if "face" in all_names and "face" not in cls_name:
                    continue

            # Integer coordinate rounding
            raw_x1, raw_y1, raw_x2, raw_y2 = map(lambda v: int(round(float(v))), xyxy)

            # Clamp coordinates strictly to image boundaries
            x1 = max(0, min(raw_x1, img_w - 1))
            y1 = max(0, min(raw_y1, img_h - 1))
            x2 = max(0, min(raw_x2, img_w))
            y2 = max(0, min(raw_y2, img_h))

            if x2 <= x1 or y2 <= y1:
                continue

            w = x2 - x1
            h = y2 - y1

            if w < min_face_size or h < min_face_size:
                continue

            # Aspect-ratio validation: reject malformed detections (slender bars, background lines)
            aspect_ratio = w / float(max(1, h))
            if aspect_ratio < 0.38 or aspect_ratio > 2.1:
                continue

            # Reject false detections that cover the entire image
            if w > 0.95 * img_w and h > 0.95 * img_h:
                continue

            tight_bbox = [x1, y1, x2, y2]
            crop_x1, crop_y1, crop_x2, crop_y2 = self.apply_padding(x1, y1, x2, y2, img_w, img_h)
            crop_bbox = [crop_x1, crop_y1, crop_x2, crop_y2]

            area = w * h
            zone = self.get_zone(area, img_w, img_h)

            # Extract 5 facial keypoints
            keypoints = []
            has_valid_kpts = False
            if keypoints_data is not None and len(keypoints_data) > i:
                kpts = keypoints_data[i].cpu().numpy()
                extracted_pts = []
                valid_count = 0
                margin_x = 0.25 * w
                margin_y = 0.25 * h
                for pt in kpts:
                    px, py = float(pt[0]), float(pt[1])
                    if not (np.isnan(px) or np.isnan(py) or np.isinf(px) or np.isinf(py)):
                        if (x1 - margin_x) <= px <= (x2 + margin_x) and (y1 - margin_y) <= py <= (y2 + margin_y):
                            valid_count += 1
                    extracted_pts.append([px, py])

                if valid_count >= 4 and len(extracted_pts) == 5:
                    keypoints = extracted_pts
                    has_valid_kpts = True

            if not has_valid_kpts:
                # Anthropometrically proportional 5 points based on the tight face box:
                # Left eye, right eye, nose, left mouth, right mouth
                keypoints = [
                    [round(x1 + w * 0.31, 1), round(y1 + h * 0.38, 1)],
                    [round(x1 + w * 0.69, 1), round(y1 + h * 0.38, 1)],
                    [round(x1 + w * 0.50, 1), round(y1 + h * 0.58, 1)],
                    [round(x1 + w * 0.34, 1), round(y1 + h * 0.78, 1)],
                    [round(x1 + w * 0.66, 1), round(y1 + h * 0.78, 1)],
                ]

            candidates.append({
                "bbox": tight_bbox,
                "crop_bbox": crop_bbox,
                "confidence": conf,
                "face_area": area,
                "zone": zone,
                "class_id": cls,
                "keypoints": keypoints
            })

        return candidates

    def _get_overlapping_tiles(self, frame_w: int, frame_h: int) -> List[Tuple[int, int, int, int]]:
        """
        Generates 2 controlled overlapping horizontal tiles covering the upper/middle classroom zone
        (where distant students sit: y in ~8% to 68% of frame height).
        Overlap is 20-25% to guarantee no student sitting at the seam is missed.
        """
        y1 = max(0, int(0.06 * frame_h))
        y2 = min(frame_h, int(0.68 * frame_h))

        # Tile 1: Left 60% of frame width
        t1_x1 = 0
        t1_x2 = min(frame_w, int(0.60 * frame_w))

        # Tile 2: Right 60% of frame width (starts at 40%, 20% overlap in middle)
        t2_x1 = max(0, int(0.40 * frame_w))
        t2_x2 = frame_w

        return [
            (t1_x1, y1, t1_x2, y2),
            (t2_x1, y1, t2_x2, y2)
        ]

    def _deduplicate_detections(
        self,
        candidate_detections: List[Dict[str, Any]],
        iou_thresh: float = 0.50,
        containment_thresh: float = 0.85
    ) -> List[Dict[str, Any]]:
        """
        Applies Secondary NMS & Containment Deduplication:
        - Prevents duplicate boxes when faces are detected in both full frame and tile crops.
        - Merges overlapping boxes by keeping the higher-confidence detection.
        - Avoids suppressing distinct adjacent students in dense classroom seating.
        """
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
                    iom = inter_area / max(1e-6, min_area) # Intersection-over-minimum

                    if iou > iou_thresh or iom > containment_thresh:
                        is_duplicate = True
                        break

            if not is_duplicate:
                final_detections.append(cand)

        return final_detections

    def detect(self, frame: Any, run_tiles: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Runs YOLOv8-Face detection with multi-scale tiling small-face recovery:
        1. Full-frame high-resolution inference (imgsz=960, conf=0.25).
        2. If tiling is enabled, runs 2 controlled overlapping crops on the distant classroom zone.
        3. Coordinates are mapped back precisely to original frame coordinates.
        4. Rigorous secondary NMS deduplication eliminates duplicate boxes.

        Args:
            frame: Input BGR image frame (numpy ndarray)
            run_tiles: True to force tiling, False to skip tiling, None to use configured schedule.

        Returns:
            List of detected faces with tight bounding boxes, confidence, keypoints, and zone.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Empty or invalid frame passed to detector.")
            return []

        t0 = time.perf_counter()
        self.frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        iou_thresh = getattr(self.settings, "nms_iou_threshold", 0.45)
        conf_thresh = getattr(self, "confidence_threshold", 0.25)
        input_size = getattr(self, "input_size", 960)
        max_faces = getattr(self.settings, "max_faces", 100)

        # 1. Full-Frame Detection
        all_candidates = self._infer_image(
            frame,
            conf_thresh=conf_thresh,
            imgsz=input_size,
            iou_thresh=iou_thresh,
            max_det=max_faces
        )

        # 2. Multi-Scale Tiling (Small-Face Recovery)
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

                tile_candidates = self._infer_image(
                    tile_crop,
                    conf_thresh=conf_thresh,
                    imgsz=640,
                    iou_thresh=iou_thresh,
                    max_det=50
                )

                # Map tile coordinates back to original full-frame space
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
                        "zone": self.get_zone(orig_w * orig_h, frame_w, frame_h),
                        "class_id": tc["class_id"],
                        "keypoints": mapped_kpts
                    })

        self.last_tile_latency_ms = (time.perf_counter() - t_tile_0) * 1000.0

        # 3. Deduplication & NMS
        final_detections = self._deduplicate_detections(all_candidates)

        self.last_detection_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latest_raw_detections = final_detections
        return final_detections

    def detect_raw(self, frame: Any) -> List[Dict[str, Any]]:
        """
        Runs YOLOv8 detection and returns raw detector boxes directly.
        """
        return self.detect(frame)


# Import SCRFDDetector
try:
    from core.scrfd_detector import SCRFDDetector
except ImportError:
    SCRFDDetector = None


def get_face_detector(model_type: Optional[str] = None, **kwargs) -> BaseDetector:
    """
    Factory function returning the configured face detector:
    Defaults to SCRFDDetector for high-accuracy multi-face detection.
    """
    settings = get_settings().detection
    target_type = (model_type or getattr(settings, "model_type", "scrfd")).lower()
    if target_type in ("yolo", "yolov8", "yolov8-face"):
        return YOLOv8FaceDetector(**kwargs)
    if SCRFDDetector is not None:
        return SCRFDDetector(**kwargs)
    return YOLOv8FaceDetector(**kwargs)


# Alias FaceDetector pointing to the default high-accuracy SCRFD detector
FaceDetector = SCRFDDetector or YOLOv8FaceDetector

