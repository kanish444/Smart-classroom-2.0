import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from loguru import logger

try:
    import lap
    HAS_LAP = True
except ImportError:
    HAS_LAP = False

from core.tracker import BaseTracker
from core.schemas import TrackState
from config.settings import get_settings


class KalmanFilterXYAH:
    """
    Standard ByteTrack Kalman Filter for tracking bounding boxes in image space.
    State: [x_center, y_center, aspect_ratio (w/h), height, vx, vy, va, vh]
    """
    def __init__(self):
        ndim = 4
        dt = 1.0

        # State transition matrix
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement projection matrix
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Standard deviation weights for noise covariance matrices
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        measurement: [x_center, y_center, a, h]
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Solve Kalman gain: K = P H^T (H P H^T + R)^(-1)
        # S = projected_cov (4x4), H P = np.dot(self._update_mat, covariance) (4x8)
        # S K^T = H P  =>  K^T = S^(-1) H P  =>  K = (S^(-1) H P)^T (8x4)
        kalman_gain = np.linalg.solve(projected_cov, np.dot(self._update_mat, covariance)).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class STrack:
    """
    Single Track entity representing a tracked face.
    """
    _count = 0

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_counter(cls):
        cls._count = 0

    def __init__(self, tlwh: np.ndarray, score: float, detection_data: Optional[Dict[str, Any]] = None):
        # [top_left_x, top_left_y, width, height]
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.detection_data = detection_data or {}
        
        self.kalman_filter: Optional[KalmanFilterXYAH] = None
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.track_id = 0
        self.state = TrackState.NEW
        self.is_activated = False

        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.last_seen = time.time()

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @property
    def xyxy(self) -> List[int]:
        box = self.tlbr
        return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

    def to_xyah(self) -> np.ndarray:
        ret = self.tlwh
        ret[:2] += ret[2:] / 2
        ret[2] /= max(1e-6, ret[3])
        return ret

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.to_xyah())

        self.tracklet_len = 0
        self.state = TrackState.NEW
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        self.last_seen = time.time()

    def mark_active(self):
        self.state = TrackState.ACTIVE
        self.is_activated = True

    def re_activate(self, new_track: "STrack", frame_id: int):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
        self.tracklet_len = 0
        self.state = TrackState.ACTIVE
        self.is_activated = True
        self.frame_id = frame_id
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.score = new_track.score
        self.detection_data = new_track.detection_data
        self.last_seen = time.time()

    def update(self, new_track: "STrack", frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.hits += 1
        self.age += 1
        self.time_since_update = 0

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
        self.score = new_track.score
        self.detection_data = new_track.detection_data
        self.last_seen = time.time()

    def predict(self):
        if self.mean is None:
            return
        mean_state = self.mean.copy()
        if self.state != TrackState.ACTIVE:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def mark_lost(self):
        self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED


def box_ious(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between two sets of boxes in tlbr [x1, y1, x2, y2] format.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.ascontiguousarray(boxes_a, dtype=np.float32)
    boxes_b = np.ascontiguousarray(boxes_b, dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    iw = np.maximum(0.0, np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T) - np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T))
    ih = np.maximum(0.0, np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T))
    intersection = iw * ih

    union = area_a[:, None] + area_b[None, :] - intersection
    ious = np.clip(intersection / np.maximum(union, 1e-6), 0.0, 1.0)
    return ious


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Solves linear sum assignment problem with threshold.
    Returns:
        matches: shape (N, 2) array of (track_idx, det_idx)
        unmatched_a: indices in A not matched
        unmatched_b: indices in B not matched
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    if HAS_LAP:
        try:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
            for ix, mx in enumerate(x):
                if mx >= 0:
                    matches.append([ix, mx])
            matches = np.asarray(matches, dtype=int) if len(matches) > 0 else np.empty((0, 2), dtype=int)
            if len(matches) > 0:
                unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matches[:, 0]]
                unmatched_b = [i for i in range(cost_matrix.shape[1]) if i not in matches[:, 1]]
            return matches, unmatched_a, unmatched_b
        except Exception as e:
            logger.debug(f"lapjv failed ({e}), falling back to greedy assignment.")
            matches = []

    # Greedy assignment fallback
    matches = []
    cost_copy = cost_matrix.copy()
    while True:
        min_val = np.min(cost_copy)
        if min_val > thresh or np.isinf(min_val):
            break
        i, j = np.unravel_index(np.argmin(cost_copy), cost_copy.shape)
        matches.append([i, j])
        cost_copy[i, :] = np.inf
        cost_copy[:, j] = np.inf

    matches = np.asarray(matches, dtype=int) if len(matches) > 0 else np.empty((0, 2), dtype=int)
    if len(matches) > 0:
        unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matches[:, 0]]
        unmatched_b = [i for i in range(cost_matrix.shape[1]) if i not in matches[:, 1]]

    return matches, unmatched_a, unmatched_b


class ByteTracker(BaseTracker):
    """
    ByteTrack implementation for multi-face tracking.
    Preserves temporary Track IDs, handles Kalman filtering, and manages TrackState lifecycle:
    NEW -> ACTIVE -> LOST -> REMOVED.
    """

    def __init__(
        self,
        track_high_thresh: Optional[float] = None,
        track_low_thresh: Optional[float] = None,
        new_track_thresh: Optional[float] = None,
        match_thresh: Optional[float] = None,
        max_lost_frames: Optional[int] = None,
        min_hits_to_activate: Optional[int] = None
    ):
        settings = get_settings().tracking
        self.track_high_thresh = track_high_thresh if track_high_thresh is not None else settings.track_high_thresh
        self.track_low_thresh = track_low_thresh if track_low_thresh is not None else settings.track_low_thresh
        self.new_track_thresh = new_track_thresh if new_track_thresh is not None else settings.new_track_thresh
        self.match_thresh = match_thresh if match_thresh is not None else settings.match_thresh
        self.max_lost_frames = max_lost_frames if max_lost_frames is not None else settings.max_lost_frames
        self.min_hits_to_activate = min_hits_to_activate if min_hits_to_activate is not None else settings.min_hits_to_activate

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYAH()
        STrack.reset_counter()

    def reset(self):
        """Reset tracker state and clear all tracks."""
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        STrack.reset_counter()

    def update(self, detections: List[Dict[str, Any]], frame: Any = None) -> List[Dict[str, Any]]:
        """
        Compatible update method returning list of track dictionaries.
        """
        active_tracks = self.update_tracks(detections, frame)
        return [
            {
                "track_id": t.track_id,
                "bbox": t.xyxy,
                "confidence": t.score,
                "state": t.state,
                "hits": t.hits,
                "age": t.age,
                "time_since_update": t.time_since_update,
                "detection_data": t.detection_data
            }
            for t in active_tracks
        ]

    def update_tracks(self, detections: List[Dict[str, Any]], frame: Any = None) -> List[STrack]:
        """
        Core ByteTrack update routine returning STrack objects.
        Executes two-stage association, lifecycle state transitions, and memory pruning.
        """
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        # 1. Parse Detections into STrack instances
        det_high: List[STrack] = []
        det_low: List[STrack] = []

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            score = float(det.get("confidence", 0.0))
            x1, y1, x2, y2 = bbox
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            tlwh = np.array([x1, y1, w, h], dtype=np.float32)

            strack = STrack(tlwh, score, detection_data=det)
            if score >= self.track_high_thresh:
                det_high.append(strack)
            elif score >= self.track_low_thresh:
                det_low.append(strack)

        # 2. Separate unconfirmed (NEW) vs confirmed (ACTIVE) tracked tracks
        unconfirmed: List[STrack] = []
        tracked_pool: List[STrack] = []

        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_pool.append(track)

        # 3. Predict Kalman state for all active and lost tracks
        strack_pool = tracked_pool + self.lost_stracks
        for track in strack_pool:
            track.predict()
            track.time_since_update += 1
            track.age += 1

        # 4. FIRST ASSOCIATION: High-confidence detections with track pool
        pool_boxes = np.array([t.tlbr for t in strack_pool]) if strack_pool else np.empty((0, 4))
        det_high_boxes = np.array([t.tlbr for t in det_high]) if det_high else np.empty((0, 4))
        cost_matrix_1 = 1.0 - box_ious(pool_boxes, det_high_boxes)

        matches_1, u_track_1, u_det_high = linear_assignment(cost_matrix_1, thresh=self.match_thresh)

        for itr, idet in matches_1:
            track = strack_pool[itr]
            det = det_high[idet]
            if track.state == TrackState.ACTIVE:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        # 5. SECOND ASSOCIATION: Low-confidence detections with remaining unmatched tracks
        unmatched_active_tracks = [strack_pool[i] for i in u_track_1 if strack_pool[i].state == TrackState.ACTIVE]
        unmatched_active_boxes = np.array([t.tlbr for t in unmatched_active_tracks]) if unmatched_active_tracks else np.empty((0, 4))
        det_low_boxes = np.array([t.tlbr for t in det_low]) if det_low else np.empty((0, 4))
        cost_matrix_2 = 1.0 - box_ious(unmatched_active_boxes, det_low_boxes)

        matches_2, u_track_2, _ = linear_assignment(cost_matrix_2, thresh=0.5)

        for itr, idet in matches_2:
            track = unmatched_active_tracks[itr]
            det = det_low[idet]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        # Tracks that failed both 1st and 2nd associations transition to LOST
        for i in u_track_2:
            track = unmatched_active_tracks[i]
            if track.state != TrackState.LOST:
                track.mark_lost()
                lost_stracks.append(track)

        # 6. UNCONFIRMED ASSOCIATION: Match unconfirmed tracks with remaining high detections
        unmatched_high_dets = [det_high[i] for i in u_det_high]
        unconfirmed_boxes = np.array([t.tlbr for t in unconfirmed]) if unconfirmed else np.empty((0, 4))
        unmatched_high_boxes = np.array([t.tlbr for t in unmatched_high_dets]) if unmatched_high_dets else np.empty((0, 4))
        cost_matrix_3 = 1.0 - box_ious(unconfirmed_boxes, unmatched_high_boxes)

        matches_3, u_unconf, u_high_remaining = linear_assignment(cost_matrix_3, thresh=0.7)

        for itr, idet in matches_3:
            track = unconfirmed[itr]
            det = unmatched_high_dets[idet]
            track.update(det, self.frame_id)
            if track.hits >= self.min_hits_to_activate:
                track.mark_active()
            activated_stracks.append(track)

        # Unconfirmed tracks not matched are removed
        for i in u_unconf:
            track = unconfirmed[i]
            track.mark_removed()
            removed_stracks.append(track)

        # 7. INITIATE NEW TRACKS: Remaining high detections become new tracks
        for i in u_high_remaining:
            det = unmatched_high_dets[i]
            if det.score >= self.new_track_thresh:
                det.activate(self.kalman_filter, self.frame_id)
                if self.min_hits_to_activate <= 1:
                    det.mark_active()
                activated_stracks.append(det)

        # 8. UPDATE LOST TRACKS: Check max_lost_frames timeout
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_lost_frames:
                track.mark_removed()
                removed_stracks.append(track)

        # 9. REORGANIZE TRACK POOLS
        new_tracked = [t for t in self.tracked_stracks if t.state == TrackState.ACTIVE]
        # Deduplicate and merge newly activated and refound tracks
        current_tracked_ids = {t.track_id for t in new_tracked}
        for t in activated_stracks + refind_stracks:
            if t.track_id not in current_tracked_ids and t.state != TrackState.REMOVED:
                new_tracked.append(t)
                current_tracked_ids.add(t.track_id)

        # Merge lost tracks
        new_lost = [t for t in self.lost_stracks if t.state == TrackState.LOST]
        current_lost_ids = {t.track_id for t in new_lost}
        for t in lost_stracks:
            if t.track_id not in current_lost_ids and t.track_id not in current_tracked_ids and t.state == TrackState.LOST:
                new_lost.append(t)
                current_lost_ids.add(t.track_id)

        # Filter out refound from lost pool
        refound_ids = {t.track_id for t in refind_stracks + activated_stracks}
        new_lost = [t for t in new_lost if t.track_id not in refound_ids]

        self.tracked_stracks = new_tracked
        self.lost_stracks = new_lost
        self.removed_stracks.extend(removed_stracks)

        # Memory cleanup: Cap removed tracks
        if len(self.removed_stracks) > 100:
            self.removed_stracks = self.removed_stracks[-50:]

        return [t for t in self.tracked_stracks if t.is_activated or t.state in (TrackState.ACTIVE, TrackState.NEW)]

