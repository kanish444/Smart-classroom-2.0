import os
from abc import ABC, abstractmethod
from typing import List, Any, Dict
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
    YOLOv8-based Face Detector.
    Uses ultralytics inference engine targeting tight face bounding boxes
    and 5 facial landmarks for ArcFace alignment.
    """
    def __init__(self, model_path: str = None):
        self.settings = get_settings().detection
        self.model_path = model_path or self.settings.model_name
        self.confidence_threshold = self.settings.confidence_threshold
        self.padding_ratio = self.settings.padding_ratio

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_dir = os.path.join(root_dir, "models", "weights")
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir, exist_ok=True)

        # Strictly prioritize the dedicated face model (yolov8n-face.pt)
        # and NEVER load a COCO person detector (yolov8n.pt) as a face detector.
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
            # Force a dry run to load model into memory
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_img, verbose=False)
            logger.info("YOLOv8 Face Detector loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

        self.latest_raw_detections: List[Dict[str, Any]] = []


    def get_zone(self, area: float, frame_width: int, frame_height: int) -> str:
        """
        Classifies the face into NEAR, MIDDLE, or FAR zone based on bounding box area
        relative to the 40x40ft classroom assumption.
        """
        if area > 15000:
            return "NEAR"
        elif area > 3000:
            return "MIDDLE"
        else:
            return "FAR"

    def apply_padding(self, x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int) -> tuple:
        """
        Applies a percentage padding to the bounding box to ensure the chin/forehead are included
        strictly for recognition crop extraction, NOT for the detection box.
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

    def detect(self, frame: Any) -> List[Dict[str, Any]]:
        """
        Runs YOLOv8 detection on the frame and extracts tight bounding boxes and keypoints.
        Guarantees:
        - Each face has its own tight bounding box [x1, y1, x2, y2]
        - Coordinates are clamped: 0 <= x1 < x2 <= frame_w and 0 <= y1 < y2 <= frame_h
        - Detection box is NOT expanded by recognition padding
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Empty or invalid frame passed to detector.")
            return []

        frame_h, frame_w = frame.shape[:2]

        iou_thresh = getattr(self.settings, "nms_iou_threshold", 0.45)
        input_size = getattr(self.settings, "input_size", 640)
        max_faces = getattr(self.settings, "max_faces", 100)
        min_face_size = getattr(self.settings, "min_face_size", 20)

        # Inference
        try:
            results = self.model(
                frame,
                verbose=False,
                conf=self.confidence_threshold,
                iou=iou_thresh,
                imgsz=input_size,
                max_det=max_faces
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []

        detections = []
        if len(results) == 0:
            return detections

        boxes = results[0].boxes
        keypoints_data = results[0].keypoints.data if results[0].keypoints is not None else None

        for i, box in enumerate(boxes):
            # Ultralytics returns tensors, move to cpu and convert to numpy/list safely
            box_xyxy = box.xyxy[0]
            if hasattr(box_xyxy, "cpu"):
                xyxy = box_xyxy.cpu().numpy().tolist()
            elif hasattr(box_xyxy, "tolist"):
                xyxy = box_xyxy.tolist()
            else:
                xyxy = list(box_xyxy)

            box_conf = box.conf[0]
            if hasattr(box_conf, "cpu"):
                conf = float(box_conf.cpu().numpy())
            else:
                conf = float(box_conf)

            box_cls = box.cls[0]
            if hasattr(box_cls, "cpu"):
                cls = int(box_cls.cpu().numpy())
            else:
                cls = int(box_cls)

            # If model defines class names and has 'face', enforce that cls corresponds to face
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                cls_name = str(self.model.names.get(cls, "")).lower()
                all_names = [str(n).lower() for n in self.model.names.values()]
                if "face" in all_names and "face" not in cls_name:
                    continue

            raw_x1, raw_y1, raw_x2, raw_y2 = map(int, xyxy)

            # Clamp coordinates to frame boundaries
            x1 = max(0, min(raw_x1, frame_w - 1))
            y1 = max(0, min(raw_y1, frame_h - 1))
            x2 = max(0, min(raw_x2, frame_w))
            y2 = max(0, min(raw_y2, frame_h))

            # Reject invalid or inverted boxes
            if x2 <= x1 or y2 <= y1:
                continue

            w = x2 - x1
            h = y2 - y1

            # Reject tiny artifact detections
            if w < min_face_size or h < min_face_size:
                continue

            # Separate detection box (tight face) from optional recognition crop padding
            tight_bbox = [x1, y1, x2, y2]
            crop_x1, crop_y1, crop_x2, crop_y2 = self.apply_padding(x1, y1, x2, y2, frame_w, frame_h)
            crop_bbox = [crop_x1, crop_y1, crop_x2, crop_y2]

            area = w * h
            zone = self.get_zone(area, frame_w, frame_h)

            # Extract 5 facial keypoints
            keypoints = []
            if keypoints_data is not None and len(keypoints_data) > i:
                kpts = keypoints_data[i].cpu().numpy()
                # kpts is typically shape [5, 3] -> (x, y, conf) or [5, 2]
                for pt in kpts:
                    keypoints.append([float(pt[0]), float(pt[1])])
            else:
                # Proportional 5 points based on the tight face box:
                # Left eye, right eye, nose, left mouth, right mouth
                cx = x1 + w / 2
                cy = y1 + h / 2
                keypoints = [
                    [x1 + w * 0.3, y1 + h * 0.4],   # left eye
                    [x1 + w * 0.7, y1 + h * 0.4],   # right eye
                    [cx, y1 + h * 0.6],             # nose
                    [x1 + w * 0.35, y1 + h * 0.8],  # left mouth
                    [x1 + w * 0.65, y1 + h * 0.8],  # right mouth
                ]

            if getattr(self.settings, "debug_mode", False):
                logger.debug(
                    f"Face #{i+1} | Box: [{x1}, {y1}, {x2}, {y2}] | W: {w} H: {h} | Conf: {conf:.2f}"
                )

            detections.append({
                "bbox": tight_bbox,
                "crop_bbox": crop_bbox,
                "confidence": conf,
                "face_area": area,
                "zone": zone,
                "class_id": cls,
                "keypoints": keypoints
            })

        self.latest_raw_detections = detections
        return detections

    def detect_raw(self, frame: Any) -> List[Dict[str, Any]]:
        """
        Runs YOLOv8 detection on the frame and returns raw detector boxes directly
        without any tracking, Kalman filtering, or identity association.
        """
        return self.detect(frame)

