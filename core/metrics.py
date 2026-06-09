from dataclasses import dataclass
from core.geometry import Box, iou

@dataclass
class SequenceMetrics:
    frames: int = 0
    target_frames: int = 0
    tracked_frames: int = 0
    true_positive_frames: int = 0
    false_positive_frames: int = 0
    lost_frames: int = 0
    reacquisitions: int = 0
    iou_sum: float = 0.0

    def update(self,
               exists: bool,
               gt_box: Box | None,
               pred_box: Box | None,
               status: str,
               event: str | None,
               iou_threshold: float) -> None:

        self.frames += 1
        if exists:
            self.target_frames += 1

        has_prediction = pred_box is not None and status in {"active", "lost"}
        if has_prediction:
            self.tracked_frames += 1

        overlap = iou(gt_box, pred_box)
        if exists and overlap >= iou_threshold:
            self.true_positive_frames += 1
            self.iou_sum += overlap
        elif not exists and status == "active":
            self.false_positive_frames += 1

        if exists and status == "lost":
            self.lost_frames += 1

        if event == "reacquired":
            self.reacquisitions += 1

    def as_dict(self) -> dict[str, float]:
        recall = self.true_positive_frames / self.target_frames if self.tracked_frames else 0.0
        fp_per_frame = self.false_positive_frames / self.frames if self.frames else 0.0
        mean_iou = self.iou_sum / self.true_positive_frames if self.true_positive_frames else 0.0
        tracked_ratio = self.tracked_frames / self.frames if self.frames else 0.0
        lost_ratio = self.lost_frames / self.target_frames if self.target_frames else 0.0

        return {
            "frames": float(self.frames),
            "target_frames": float(self.target_frames),
            "recall": recall,
            "false_positive_per_frame": fp_per_frame,
            "mean_iou_on_hits": mean_iou,
            "tracked_frame_ratio": tracked_ratio,
            "lost_frame_ratio": lost_ratio,
            "reacquisitions": float(self.reacquisitions)
        }