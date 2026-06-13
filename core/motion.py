from dataclasses import dataclass

import cv2
import numpy as np

from core.geometry import Box, box_area, box_center, clip_box, expand_box, iou


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
        max_area: int = 9000,
        pad: int = 6,
        warmup_frames: int = 3,
        merge_scale: float = 2.0,
        proposal_scale: float = 1.35,
        max_box_width: int = 260,
        max_box_height: int = 190,
        debug: bool = False,
        merge_margin: int = 15,
        min_aspect_ratio: float = 0.2,
        max_aspect_ratio: float = 5.0
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
        self.merge_scale = merge_scale
        self.proposal_scale = proposal_scale
        self.max_box_width = max_box_width
        self.max_box_height = max_box_height
        self.debug = debug
        self.frame_count = 0
        self.merge_margin = merge_margin
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def detect(self, frame: np.ndarray, roi: Box | None = None) -> list[Detection]:
        height, width = frame.shape[:2]
        x_offset, y_offset = 0, 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        mask = self.bg.apply(gray)
        self.frame_count += 1

        if self.frame_count <= self.warmup_frames:
            return []

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        search_mask = mask
        if roi is not None:
            x, y, w, h = clip_box(roi, width, height)
            x_offset, y_offset = int(x), int(y)
            search_mask = mask[int(y) : int(y + h), int(x) : int(x + w)]

        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            raw_boxes.append((cx, cy, cw, ch))

        if not raw_boxes:
            return []

        cluster_mask = np.zeros_like(search_mask)
        for cx, cy,cw ,ch in raw_boxes:
            ex = max(0, cx - self.merge_margin)
            ey = max(0, cy - self.merge_margin)
            ex2 = min(search_mask.shape[1], cx+cw+self.merge_margin)
            ey2 = min(search_mask.shape[0], cy+ch+self.merge_margin)
            cv2.rectangle(cluster_mask, (ex, ey), (ex2, ey2), 255, -1)

        cluster_contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clusters = {i: [] for i in range(len(cluster_contours))}
        for rx, ry, rw, rh in raw_boxes:
            center_x = rx + rw / 2.0
            center_y = ry + rh / 2.0
            for i, contour in enumerate(cluster_contours):
                ccx, ccy, ccw, cch = cv2.boundingRect(contour)
                if ccx <= center_x <= ccx + ccw and ccy <= center_y <= ccy + cch:
                    clusters[i].append((rx, ry, rw, rh))
                    break


        detections: list[Detection] = []
        for i, boxes in clusters.items():
            if not boxes:
                continue

            min_x = min(b[0] for b in boxes)
            min_y = min(b[1] for b in boxes)
            max_x = max(b[0] + b[2] for b in boxes)
            max_y = max(b[1] + b[3] for b in boxes)

            bw = max_x - min_x
            bh = max_y - min_y

            aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            bx = min_x + x_offset - self.pad
            by = min_y + y_offset - self.pad
            bw = bw + self.pad * 2
            bh = bh + self.pad * 2

            box = clip_box((float(bx), float(by), float(bw), float(bh)), width, height)
            area = box_area(box)

            # if area < self.min_area or area > self.max_area:
            #     continue

            score = min(1.0, area / float(self.max_area))
            detections.append(Detection(box=box, score=score))

        if self.debug:
            self._show_debug(search_mask, contours, detections, x_offset, y_offset)

        detections.sort(key=lambda item: item.score, reverse=True)
        debug_frame = frame.copy()

        # for det in detections:
        #     # Kutunun koordinatlarını integer'a çevir (x, y, w, h)
        #     dx, dy, dw, dh = [int(v) for v in det.box]
        #
        #     # Bulunan hareketli kümeyi sarı renkte çiz
        #     cv2.rectangle(debug_frame, (dx, dy), (dx + dw, dy + dh), (0, 255, 255), 2)
        #
        #     # Kutunun üzerine algılanma skorunu yaz (opsiyonel)
        #     cv2.putText(debug_frame, f"{det.score:.2f}", (dx, max(0, dy - 5)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        #
        # # Ekranda göster
        # cv2.imshow("Motion Detector - Cluster Results", debug_frame)
        #
        # # Eğer videonun akmasını istiyorsan 1,
        # # her karede durup manuel olarak Space/Enter ile ilerlemek istiyorsan 0 yap.
        # cv2.waitKey(0)

        return detections

    def _build_object_proposals(
        self,
        components: list[tuple[Box, float]],
        frame_width: int,
        frame_height: int,
    ) -> list[Detection]:
        if not components:
            return []

        groups = self._cluster_components([box for box, _ in components], frame_width, frame_height)
        proposals: list[Detection] = []
        for group in groups:
            group_area = sum(components[index][1] for index in group)
            union = self._union_boxes([components[index][0] for index in group])
            proposal_box = expand_box(union, self.proposal_scale, frame_width, frame_height)
            if not self._is_valid_proposal(proposal_box):
                continue

            score = self._proposal_score(proposal_box, group_area, len(group))
            proposals.append(Detection(box=proposal_box, score=score, source="motion_group"))

        return self._suppress_contained_boxes(proposals)

    def _cluster_components(
        self,
        boxes: list[Box],
        frame_width: int,
        frame_height: int,
    ) -> list[list[int]]:
        parent = list(range(len(boxes)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        expanded = [expand_box(box, self.merge_scale, frame_width, frame_height) for box in boxes]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if self._boxes_should_merge(boxes[i], boxes[j], expanded[i], expanded[j]):
                    merged = self._union_boxes([boxes[i], boxes[j]])
                    if self._is_reasonable_cluster_size(merged):
                        union(i, j)

        groups_by_root: dict[int, list[int]] = {}
        for index in range(len(boxes)):
            groups_by_root.setdefault(find(index), []).append(index)
        return list(groups_by_root.values())

    def _boxes_should_merge(self, a: Box, b: Box, expanded_a: Box, expanded_b: Box) -> bool:
        if iou(expanded_a, expanded_b) > 0.0:
            return True

        ax, ay = box_center(a)
        bx, by = box_center(b)
        gap_x = max(0.0, abs(ax - bx) - (a[2] + b[2]) * 0.5)
        gap_y = max(0.0, abs(ay - by) - (a[3] + b[3]) * 0.5)
        max_gap_x = max(18.0, min(a[2], b[2]) * 1.2)
        max_gap_y = max(14.0, min(a[3], b[3]) * 1.0)
        return gap_x <= max_gap_x and gap_y <= max_gap_y

    def _is_reasonable_cluster_size(self, box: Box) -> bool:
        return box[2] <= self.max_box_width and box[3] <= self.max_box_height

    def _is_valid_proposal(self, box: Box) -> bool:
        width = box[2]
        height = box[3]
        area = box_area(box)
        if area < 20.0 or area > float(self.max_area):
            return False
        if width > self.max_box_width or height > self.max_box_height:
            return False

        aspect = width / max(height, 1.0)
        return 0.20 <= aspect <= 6.0

    def _proposal_score(self, box: Box, motion_area: float, component_count: int) -> float:
        area = max(1.0, box_area(box))
        fill_ratio = min(1.0, motion_area / area)
        size_score = min(1.0, area / 4500.0)
        group_bonus = min(0.18, max(0, component_count - 1) * 0.06)
        return float(np.clip(0.30 + size_score * 0.35 + fill_ratio * 0.25 + group_bonus, 0.0, 1.0))

    def _suppress_contained_boxes(self, detections: list[Detection]) -> list[Detection]:
        ordered = sorted(detections, key=lambda detection: (detection.score, box_area(detection.box)), reverse=True)
        kept: list[Detection] = []
        for detection in ordered:
            if any(self._contained_overlap(detection.box, kept_detection.box) > 0.82 for kept_detection in kept):
                continue
            kept.append(detection)
        return kept

    def _contained_overlap(self, a: Box, b: Box) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return inter / max(1.0, min(box_area(a), box_area(b)))

    def _union_boxes(self, boxes: list[Box]) -> Box:
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[0] + box[2] for box in boxes)
        y2 = max(box[1] + box[3] for box in boxes)
        return x1, y1, x2 - x1, y2 - y1

    def _show_debug(
        self,
        search_mask: np.ndarray,
        contours,
        detections: list[Detection],
        x_offset: int,
        y_offset: int,
    ) -> None:
        debug_mask = cv2.cvtColor(search_mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_mask, contours, -1, (0, 255, 0), 1)
        # for detection in detections:
        #     x, y, w, h = detection.box
        #     x = int(x - x_offset)
        #     y = int(y - y_offset)
        #     cv2.rectangle(debug_mask, (x, y), (x + int(w), y + int(h)), (0, 0, 255), 1)
        #
        # cv2.imshow("Debug: Mask Contours", debug_mask)
        # cv2.waitKey(1)

def detections_in_roi(detections: list[Detection], roi: Box | None) -> list[Detection]:
    if roi is None:
        return []
    return [det for det in detections if iou(det.box, roi) > 0.0]
