from dataclasses import dataclass

from core.geometry import Box, center_distance, expand_box, iou
from core.kalman import KalmanBoxTracker
from core.motion import Detection

@dataclass
class TrackState:
    track_id: int
    box: Box | None
    status: str
    age: int
    hits: int
    missed: int
    uncertainty: float
    event: str | None = None


class OcclusionAwareSingleTracker:
    def __init__(self,
                 max_lost: int = 45,
                 min_iou: float = 0.02,
                 max_center_distance: float = 90.0,
                 roi_scale: float = 5.0):

        self.max_lost = max_lost
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.roi_scale = roi_scale
        self.missed = 0
        self.hits = 0
        self.kalman = KalmanBoxTracker()
        self.age = 0
        self.track_id = 1
        self.status = "empty"

    def search_roi(self, frame_width: int, frame_height: int) -> Box | None:
        predicted = self.kalman.current_box() if self.kalman.initialized else None
        if predicted is None or self.status == "empty":
            return None
        scale = self.roi_scale + min(self.missed, self.max_lost) * 0.08
        return expand_box(predicted, scale, frame_width, frame_height)

    def update(self, detections: list[Detection]) -> TrackState:
        predicted = self.kalman.predict()
        self.age += 1

        match = self._best_match(predicted, detections)
        event = None

        if match is not None:
            was_lost = self.status == "lost"
            self.kalman.update(match.box)
            self.status = "active"
            self.hits += 1
            self.missed = 0
            event = "reacquired" if was_lost else "matched"
        elif self.kalman.initialized:
            self.missed += 1
            self.status = "lost" if self.missed <= self.max_lost else "terminated"
            event = "lost" if self.status == "lost" else "terminated"
        else:
            self.status = "empty"

        return TrackState(
            track_id=self.track_id,
            box=self.kalman.current_box() if self.kalman.initialized else None,
            status=self.status,
            age=self.age,
            hits=self.hits,
            missed=self.missed,
            uncertainty=self.kalman.uncertainty() if self.kalman.initialized else 0.0,
            event=event
        )

    def _best_match(self, predicted: Box | None, detections: list[Detection]) -> Detection | None:
        if not detections:
            return None

        if predicted is None or self.status == "empty":
            return detections[0]

        best_detection = None
        best_score = float("-inf")

        for detection in detections:
            overlap = iou(predicted, detection.box)
            distance = center_distance(predicted, detection.box)
            distance_score = max(0.0, 1.0 - distance/self.max_center_distance)
            score = overlap * 0.7 + distance_score * 0.3 + detection.score * 0.1
            if score > best_score:
                best_detection = detection
                best_score = score

        if best_detection is None:
            return None

        overlap = iou(predicted, best_detection.box)
        distance = center_distance(predicted, best_detection.box)
        if overlap >= self.min_iou or distance <= self.max_center_distance:
            return best_detection

        return None