import math
from dataclasses import dataclass

import cv2
import numpy as np

from core.geometry import Box, box_area, center_distance, clip_box, expand_box, iou
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
                 roi_scale: float = 2.5,
                 reacquire_min_score: float = 0.58,
                 reacquire_min_detection_score: float = 0.50,
                 appearance_update_rate: float = 0.12,
                 debug: bool = False):

        self.max_lost = max_lost
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.roi_scale = roi_scale
        self.reacquire_min_score = reacquire_min_score
        self.reacquire_min_detection_score = reacquire_min_detection_score
        self.appearance_update_rate = appearance_update_rate
        self.debug = debug
        self.missed = 0
        self.hits = 0
        self.kalman = KalmanBoxTracker()
        self.age = 0
        self.track_id = 1
        self.status = "empty"
        self.last_observed_box: Box | None = None
        self.appearance_hist: np.ndarray | None = None

    def search_roi(self, frame_width: int, frame_height: int) -> Box | None:
        predicted = self.kalman.predicted_box() if self.kalman.initialized else None
        if predicted is None or self.status == "empty":
            return None
        scale = self.roi_scale + min(self.missed, self.max_lost) * 0.08
        return expand_box(predicted, scale, frame_width, frame_height)

    def update(
        self,
        primary_detections: list[Detection],
        fallback_detections: list[Detection] | None = None,
        frame: np.ndarray | None = None,
    ) -> TrackState:
        predicted = self.kalman.predict()
        self.age += 1

        match = self._best_match(predicted, primary_detections)
        match_source = "primary"
        if match is None and fallback_detections is not None:
            match = self._best_reacquire_match(predicted, fallback_detections, frame)
            match_source = "fallback"
        event = None

        if match is not None:
            was_lost = self.status == "lost"
            self.kalman.update(match.box)
            self.status = "active"
            self.hits += 1
            self.missed = 0
            self.last_observed_box = match.box
            self._update_appearance(frame, match.box)
            if was_lost:
                event = "reacquired"
            elif match_source == "fallback":
                event = "global_match"
            else:
                event = "matched"
        elif self.kalman.initialized:
            self.missed += 1
            self.status = "lost" if self.missed <= self.max_lost else "terminated"
            event = "lost" if self.status == "lost" else "terminated"
        else:
            self.status = "empty"

        if frame is not None:
            # 1. Kalman'ın Tahmini (PREDICTED) -> MAVİ ÇİZELİM
            # (Filtrenin nesneyi nerede beklediğini gösterir)
            # if predicted is not None:
            #     # Not: Box formatının (x, y, w, h) olduğunu varsayıyoruz.
            #     # Eğer yapınız (x1, y1, x2, y2) ise: px, py, px2, py2 olarak değiştirin.
            #     px, py, pw, ph = map(int, predicted)
            #     cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 0, 0), 2)
            #     cv2.putText(frame, f"Pred ID:{self.track_id}", (px, max(0, py - 5)),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 2. Gerçekleşen Eşleşme (MATCH) -> YEŞİL ÇİZELİM
            # (Algılamadan gelen gerçek sonuç)
            if match is not None:
                mx, my, mw, mh = map(int, match.box)
                cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
                cv2.putText(frame, f"Match ID:{self.track_id}", (mx, my + mh + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Debug: Tracker Output", frame)
            # Eğer görüntüyü dondurup adım adım gitmek istersen 1 yerine 0 yaz.
            # Kodun takılmasını engellemek için waitKey şarttır.
            cv2.waitKey(1)


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

    def _best_reacquire_match(
        self,
        predicted: Box | None,
        detections: list[Detection],
        frame: np.ndarray | None,
    ) -> Detection | None:
        if not detections:
            return None

        uses_cnn_scores = any("patch" in detection.source for detection in detections)
        min_detection_score = self.reacquire_min_detection_score if uses_cnn_scores else 0.0

        best_detection = None
        best_score = float("-inf")
        for detection in detections:
            if detection.score < min_detection_score:
                continue

            score = self._reacquire_score(predicted, detection, frame)
            if score > best_score:
                best_detection = detection
                best_score = score

        if best_detection is None:
            return None

        if predicted is None or self.status == "empty":
            return best_detection

        if best_score >= self._reacquire_threshold():
            return best_detection

        return None

    def _reacquire_threshold(self) -> float:
        if self.status == "lost":
            return max(0.50, self.reacquire_min_score - min(self.missed, 20) * 0.006)
        return self.reacquire_min_score

    def _reacquire_score(
        self,
        predicted: Box | None,
        detection: Detection,
        frame: np.ndarray | None,
    ) -> float:
        detection_score = float(detection.score)
        motion_score = self._motion_compatibility(predicted, detection.box)
        size_score = self._size_compatibility(predicted, detection.box)
        appearance_score = self._appearance_similarity(frame, detection.box)

        return (
            detection_score * 0.35
            + motion_score * 0.25
            + size_score * 0.20
            + appearance_score * 0.20
        )

    def _motion_compatibility(self, predicted: Box | None, box: Box) -> float:
        if predicted is None:
            return 0.50

        distance = center_distance(predicted, box)
        uncertainty_radius = math.sqrt(max(0.0, self.kalman.uncertainty())) * 0.25
        missed_scale = 1.0 + min(self.missed, self.max_lost) * 0.20
        radius = max(self.max_center_distance, self.max_center_distance * missed_scale + uncertainty_radius)
        return float(math.exp(-0.5 * (distance / max(radius, 1.0)) ** 2))

    def _size_compatibility(self, predicted: Box | None, box: Box) -> float:
        reference = self.last_observed_box or predicted
        if reference is None:
            return 0.50

        reference_area = max(1.0, box_area(reference))
        candidate_area = max(1.0, box_area(box))
        area_score = math.exp(-abs(math.log(candidate_area / reference_area)))

        reference_aspect = max(1e-6, reference[2] / max(reference[3], 1e-6))
        candidate_aspect = max(1e-6, box[2] / max(box[3], 1e-6))
        aspect_score = math.exp(-abs(math.log(candidate_aspect / reference_aspect)))
        return float(area_score * 0.65 + aspect_score * 0.35)

    def _appearance_similarity(self, frame: np.ndarray | None, box: Box) -> float:
        if frame is None or self.appearance_hist is None:
            return 0.50

        hist = self._appearance_histogram(frame, box)
        if hist is None:
            return 0.50

        similarity = cv2.compareHist(
            self.appearance_hist.astype(np.float32),
            hist.astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        return float(np.clip((similarity + 1.0) * 0.5, 0.0, 1.0))

    def _update_appearance(self, frame: np.ndarray | None, box: Box) -> None:
        if frame is None:
            return

        hist = self._appearance_histogram(frame, box)
        if hist is None:
            return

        if self.appearance_hist is None:
            self.appearance_hist = hist
            return

        rate = self.appearance_update_rate
        updated = (1.0 - rate) * self.appearance_hist + rate * hist
        norm = np.linalg.norm(updated)
        self.appearance_hist = updated / norm if norm > 0.0 else updated

    def _appearance_histogram(self, frame: np.ndarray, box: Box) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x, y, w, h = clip_box(expand_box(box, 1.8, width, height), width, height)
        crop = frame[int(y):int(y + h), int(x):int(x + w)]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_LINEAR)
        if crop.ndim == 2:
            hist = cv2.calcHist([crop], [0], None, [32], [0, 256])
        else:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])

        hist = hist.astype(np.float32).flatten()
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0.0 else hist
