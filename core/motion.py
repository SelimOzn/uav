from dataclasses import dataclass

import cv2
import numpy as np

from core.geometry import Box, box_area, clip_box


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float
    source: str = "motion"


class MotionCandidateDetector:
    def __init__(
        self,
        history: int = 80,
        var_threshold: float = 18.0,
        min_area: int = 4,
        max_area: int = 3000,
        pad: int = 4,
        warmup_frames: int = 3,
    ):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self.min_area = min_area
        self.max_area = max_area
        self.pad = pad
        self.warmup_frames = warmup_frames
        self.frame_count = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray, roi: Box | None = None) -> list[Detection]:
        height, width = frame.shape[:2]
        x_offset, y_offset = 0, 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        mask = self.bg.apply(gray)
        self.frame_count += 1

        if self.frame_count <= self.warmup_frames:
            return []

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        search_mask = mask
        if roi is not None:
            x, y, w, h = clip_box(roi, width, height)
            x_offset, y_offset = int(x), int(y)
            search_mask = mask[int(y) : int(y + h), int(x) : int(x + w)]

        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            x = max(0, x - self.pad) + x_offset
            y = max(0, y - self.pad) + y_offset
            w = w + self.pad * 2
            h = h + self.pad * 2
            box = clip_box((float(x), float(y), float(w), float(h)), width, height)
            score = min(1.0, box_area(box) / float(self.max_area))
            detections.append(Detection(box=box, score=score))

        detections.sort(key=lambda item: item.score, reverse=True)
        return detections
