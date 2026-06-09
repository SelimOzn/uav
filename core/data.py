import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2

@dataclass(frozen=True)
class FrameRecord:
    frame_idx: int
    image: object
    exists: bool
    gt_box: tuple[float, float, float, float] | None

@dataclass(frozen=True)
class AntiUAVSequence:
    name: str
    video_path: Path
    annotation_path: Path

    @classmethod
    def from_dir(cls, sequence_dir: str | Path, modality: str = "visible") -> "AntiUAVSequence":
        sequence_dir = Path(sequence_dir)
        video_path = sequence_dir / f"{modality}.mp4"
        annotation_path = sequence_dir / f"{modality}.json"

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation not found: {annotation_path}")

        return cls(sequence_dir.name, video_path, annotation_path)

    def load_annotations(self) -> dict:
        with self.annotation_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def iter_frames(self, max_frames: int | None = None) -> Iterator[FrameRecord]:
        annotations = self.load_annotations()
        exists = annotations["exist"]
        gt_rects = annotations["gt_rect"]

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {self.video_path}")

        frame_idx = 0
        len_exists = len(exists)
        len_gt_rects = len(gt_rects)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break

                has_target = bool(exists[frame_idx]) if frame_idx < len_exists else False
                gt_box = None
                if has_target and frame_idx < len_gt_rects:
                    rect = gt_rects[frame_idx]
                    if len(rect) == 4:
                        gt_box = tuple(float(v) for v in rect)

                yield FrameRecord(frame_idx, frame, has_target, gt_box)
                frame_idx += 1
        finally:
            cap.release()
