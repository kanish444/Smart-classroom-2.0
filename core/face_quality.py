import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from core.schemas import FaceQualityResult
from config.settings import get_settings


class FaceQualityAssessor:
    """
    Evaluates raw bounding box crops for sufficient quality (resolution, sharpness, brightness,
    contrast, glare, edge-clipping) before they are passed to alignment and ArcFace recognition.

    Produces granular states:
    - RECOGNITION_READY: Passed all resolution, sharpness, illumination, and landmark checks.
    - FACE_TOO_SMALL: Detected face has insufficient pixels (< 36x36 px or < 1600 px area)
      for reliable ArcFace 112x112 upsampling; recognition must NOT be attempted.
    - LOW_QUALITY: Blurry, overexposed/glare, too dark, low contrast, or partially visible at frame boundary.
    """

    def __init__(self):
        self.settings = get_settings().quality

    def evaluate_sharpness(self, gray: np.ndarray) -> float:
        """Calculates Variance of the Laplacian. Higher indicates sharper features."""
        if gray is None or gray.size == 0:
            return 0.0
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def evaluate_brightness(self, gray: np.ndarray) -> float:
        """Calculates mean pixel intensity of the crop (0=black, 255=white)."""
        if gray is None or gray.size == 0:
            return 0.0
        return float(np.mean(gray))

    def evaluate_contrast(self, gray: np.ndarray) -> float:
        """Calculates luminance standard deviation (RMS contrast). Low values indicate washed-out lighting."""
        if gray is None or gray.size == 0:
            return 0.0
        return float(np.std(gray))

    def evaluate_glare(self, gray: np.ndarray) -> float:
        """Calculates fraction of overexposed/saturated pixels (> 240 intensity)."""
        if gray is None or gray.size == 0:
            return 0.0
        return float(np.mean(gray > 240))

    def evaluate_pose(self, keypoints: Optional[List[List[float]]]) -> Tuple[str, float, float, float]:
        """
        Estimates pose tilt angle and yaw offset using 5 facial keypoints (adapted from reference project).
        Returns: (pose_label, tilt_angle, yaw_offset, pose_score)
        """
        if not keypoints or len(keypoints) < 5:
            return "UNKNOWN", 0.0, 0.0, 0.8

        try:
            left_eye = np.array(keypoints[0], dtype=np.float32)
            right_eye = np.array(keypoints[1], dtype=np.float32)
            nose = np.array(keypoints[2], dtype=np.float32)

            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            eye_dist = float(np.hypot(dx, dy))

            if eye_dist > 5.0:
                tilt_angle = float(abs(np.degrees(np.arctan2(dy, dx))))
                mid_eye = (left_eye + right_eye) / 2.0
                yaw_offset = float(abs(nose[0] - mid_eye[0]) / (eye_dist * 0.5 + 1e-6))

                tilt_penalty = min(1.0, tilt_angle / 50.0)
                yaw_penalty = min(1.0, yaw_offset)
                pose_score = max(0.0, 1.0 - (0.4 * tilt_penalty + 0.6 * yaw_penalty))

                max_tilt = getattr(self.settings, "max_tilt_angle", 35.0)
                max_yaw = getattr(self.settings, "max_yaw_offset", 0.80)

                if tilt_angle > max_tilt or yaw_offset > max_yaw:
                    pose_label = "EXTREME_POSE"
                elif tilt_angle > 20.0 or yaw_offset > 0.50:
                    pose_label = "SLIGHT_ROTATION"
                else:
                    pose_label = "FRONTAL"

                return pose_label, round(tilt_angle, 1), round(yaw_offset, 2), round(pose_score, 3)
        except Exception:
            pass

        return "FRONTAL", 0.0, 0.0, 1.0

    def compute_composite_score(
        self,
        sharpness: float,
        brightness: float,
        contrast: float,
        w: int,
        h: int,
        pose_score: float
    ) -> float:
        """
        Computes normalized composite quality score [0.0, 1.0] (adapted from reference project)
        combining sharpness, brightness, contrast, size, and pose.
        """
        sharp_norm = min(1.0, max(0.0, sharpness / 150.0))

        if brightness < 30.0:
            bright_norm = max(0.0, brightness / 60.0)
        elif brightness > 230.0:
            bright_norm = max(0.0, 1.0 - (brightness - 230.0) / 25.0)
        else:
            bright_norm = 1.0 - abs(brightness - 128.0) / 140.0

        contrast_norm = min(1.0, max(0.0, contrast / 50.0))

        min_dim = getattr(self.settings, "min_face_dimension", 36)
        eff_dim = min(w, h)
        if eff_dim < min_dim:
            size_norm = max(0.1, (eff_dim / float(max(1, min_dim))) * 0.4)
        else:
            size_norm = min(1.0, 0.6 + 0.4 * ((eff_dim - min_dim) / 120.0))

        composite = (
            0.25 * sharp_norm +
            0.20 * bright_norm +
            0.20 * size_norm +
            0.20 * pose_score +
            0.15 * contrast_norm
        )
        return round(float(np.clip(composite, 0.0, 1.0)), 3)

    def assess(self, face_dict: Dict[str, Any], frame: np.ndarray, face_id: str) -> FaceQualityResult:
        """
        Assesses a single detection dictionary and returns the structured FaceQualityResult.
        Guarantees safe bounds checking and produces granular quality states:
        - GOOD
        - LOW QUALITY
        - FACE TOO SMALL
        - TOO BLURRY
        - TOO DARK
        - OVEREXPOSED
        - EXTREME POSE
        - PARTIAL
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return FaceQualityResult(
                face_id=face_id,
                bbox=[0, 0, 0, 0],
                keypoints=None,
                width=0,
                height=0,
                area=0,
                sharpness=0.0,
                brightness=0.0,
                contrast=0.0,
                glare_score=0.0,
                pose="UNKNOWN",
                quality_status="LOW_QUALITY",
                quality_state="LOW QUALITY",
                quality_score=0.0,
                is_acceptable=False,
                rejection_reason="Invalid or empty image frame"
            )

        frame_h, frame_w = frame.shape[:2]
        bbox = face_dict.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox

        # Check if bounding box is completely outside frame boundaries
        if x1 >= frame_w or y1 >= frame_h or x2 <= 0 or y2 <= 0:
            return FaceQualityResult(
                face_id=face_id,
                bbox=[x1, y1, x2, y2],
                keypoints=face_dict.get('keypoints'),
                width=0,
                height=0,
                area=0,
                sharpness=0.0,
                brightness=0.0,
                contrast=0.0,
                glare_score=0.0,
                pose="UNKNOWN",
                quality_status="LOW_QUALITY",
                quality_state="LOW QUALITY",
                quality_score=0.0,
                is_acceptable=False,
                rejection_reason="Bounding box completely outside frame boundaries"
            )

        # Ensure safe frame bounds
        x1_safe = max(0, min(x1, frame_w - 1))
        y1_safe = max(0, min(y1, frame_h - 1))
        x2_safe = max(x1_safe + 1, min(x2, frame_w))
        y2_safe = max(y1_safe + 1, min(y2, frame_h))

        w = x2_safe - x1_safe
        h = y2_safe - y1_safe
        area = w * h

        if w <= 4 or h <= 4:
            return FaceQualityResult(
                face_id=face_id,
                bbox=[x1_safe, y1_safe, x2_safe, y2_safe],
                keypoints=face_dict.get('keypoints'),
                width=w,
                height=h,
                area=area,
                sharpness=0.0,
                brightness=0.0,
                contrast=0.0,
                glare_score=0.0,
                pose="UNKNOWN",
                quality_status="LOW_QUALITY",
                quality_state="LOW QUALITY",
                quality_score=0.0,
                is_acceptable=False,
                rejection_reason="Too small (degenerate crop area)"
            )

        crop = frame[y1_safe:y2_safe, x1_safe:x2_safe]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Calculate Quality & Glare Diagnostic Metrics
        sharpness = self.evaluate_sharpness(gray)
        brightness = self.evaluate_brightness(gray)
        contrast = self.evaluate_contrast(gray)
        glare_score = self.evaluate_glare(gray)

        # Pose evaluation from 5 facial keypoints
        keypoints = face_dict.get('keypoints')
        pose_label, tilt_angle, yaw_offset, pose_score = self.evaluate_pose(keypoints)

        # Composite quality score
        composite_score = self.compute_composite_score(
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            w=w,
            h=h,
            pose_score=pose_score
        )

        # Thresholds from settings
        min_area = getattr(self.settings, "min_face_area", 1600)
        min_dim = getattr(self.settings, "min_face_dimension", 36)
        min_sharp = getattr(self.settings, "min_sharpness", 35.0)
        min_bright = getattr(self.settings, "min_brightness", 30.0)
        max_bright = getattr(self.settings, "max_brightness", 230.0)
        min_contrast = getattr(self.settings, "min_contrast", 12.0)
        max_glare = getattr(self.settings, "max_glare_ratio", 0.40)
        max_tilt = getattr(self.settings, "max_tilt_angle", 35.0)
        max_yaw = getattr(self.settings, "max_yaw_offset", 0.80)

        # Quality Gating Logic (Evaluates granular states: GOOD, LOW QUALITY, FACE TOO SMALL,
        # TOO BLURRY, TOO DARK, OVEREXPOSED, EXTREME POSE, PARTIAL)
        is_on_edge = (x1_safe <= 2 or x2_safe >= frame_w - 2 or y1_safe <= 2 or y2_safe >= frame_h - 2)
        aspect_ratio = w / float(max(1, h))

        if is_on_edge and (aspect_ratio < 0.50 or aspect_ratio > 2.0):
            quality_state = "PARTIAL"
            reason = f"Partially visible face at frame boundary (aspect ratio: {aspect_ratio:.2f})"
        elif area < min_area or min(w, h) < min_dim:
            quality_state = "FACE TOO SMALL"
            reason = f"Too small ({w}x{h} px, area {area} < {min_area} px)"
        elif glare_score > max_glare or brightness > max_bright:
            quality_state = "OVEREXPOSED"
            reason = f"Too bright / Glare (brightness {brightness:.1f} > {max_bright}, glare {glare_score:.2f})"
        elif brightness < min_bright:
            quality_state = "TOO DARK"
            reason = f"Too dark ({brightness:.1f} < {min_bright})"
        elif sharpness < min_sharp:
            quality_state = "TOO BLURRY"
            reason = f"Too blurry ({sharpness:.1f} < {min_sharp})"
        elif pose_label == "EXTREME_POSE" or tilt_angle > max_tilt or yaw_offset > max_yaw:
            quality_state = "EXTREME POSE"
            reason = f"Extreme pose angle (tilt {tilt_angle:.1f} deg > {max_tilt}, yaw {yaw_offset:.2f} > {max_yaw})"
        elif contrast < min_contrast:
            quality_state = "LOW QUALITY"
            reason = f"Low facial contrast ({contrast:.1f} < {min_contrast})"
        else:
            quality_state = "GOOD"
            reason = None

        is_acceptable = (quality_state == "GOOD")
        quality_status = "RECOGNITION_READY" if is_acceptable else "LOW_QUALITY"

        return FaceQualityResult(
            face_id=face_id,
            bbox=[x1_safe, y1_safe, x2_safe, y2_safe],
            keypoints=keypoints,
            width=w,
            height=h,
            area=area,
            sharpness=round(sharpness, 1),
            brightness=round(brightness, 1),
            contrast=round(contrast, 1),
            glare_score=round(glare_score, 3),
            pose=pose_label,
            quality_status=quality_status,
            quality_state=quality_state,
            quality_score=composite_score,
            is_acceptable=is_acceptable,
            tilt_angle=tilt_angle,
            yaw_offset=yaw_offset,
            rejection_reason=reason
        )
