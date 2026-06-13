import argparse
import time
from pathlib import Path

import cv2

from core.data import AntiUAVSequence
from core.metrics import SequenceMetrics
from core.motion import MotionCandidateDetector, detections_in_roi
from core.tracker import OcclusionAwareSingleTracker

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU baseline for tiny UAV tracking.")
    parser.add_argument("--sequence", type=Path, required=True, help="Path to an Anti-UAV sequence folder.")
    parser.add_argument("--modality", default="visible", choices=["visible", "infrared"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--write-video", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="Patch classifier device: auto, cuda, or cpu.")
    parser.add_argument("--patch-model", type=Path, default=Path().cwd() / "weights/tiny_patch_cnn.pt", help="Optional tiny patch CNN checkpoint.", )
    parser.add_argument("--patch-threshold", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=20, help="Keep top K patch-classified detections per frame.")
    return parser.parse_args()

def draw_overlay(frame, gt_box, pred_box, status: str, event: str | None):
    if gt_box is not None:
        x, y, w, h = [int(v) for v in gt_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

    if pred_box is not None:
        x, y, w, h = [int(v) for v in pred_box]
        color = (0, 0, 255) if status == "lost" else (0, 180, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.putText(
        frame,
        f"{status} {event or ''}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame

def main() -> None:
    args = parse_args()
    sequence = AntiUAVSequence.from_dir(args.sequence, args.modality)
    detector = MotionCandidateDetector()
    tracker = OcclusionAwareSingleTracker()
    metrics = SequenceMetrics()
    patch_classifier = None

    if args.patch_model is not None:
        from core.patch_classifier import PatchClassifier

        patch_classifier = PatchClassifier(
            checkpoint_path=args.patch_model,
            device=args.device,
            threshold=args.patch_threshold,
        )
        print(f"patch_classifier_device: {patch_classifier.device}")

    writer = None
    start = time.perf_counter()

    for record in sequence.iter_frames(args.max_frames):
        height, width = record.image.shape[:2]
        all_detections = detector.detect(record.image, roi=None)
        roi = tracker.search_roi(width, height)
        if patch_classifier is not None:
            global_cnn_detections = patch_classifier.filter(record.image, all_detections, top_k=args.top_k)
        else:
            global_cnn_detections = all_detections
        local_detections = detections_in_roi(global_cnn_detections, roi)

        state = tracker.update(
            primary_detections=local_detections,
            fallback_detections=global_cnn_detections,
            frame=record.image,
        )

        metrics.update(
            exists=record.exists,
            gt_box=record.gt_box,
            pred_box=state.box,
            status=state.status,
            event=state.event,
            iou_threshold=args.iou_threshold,
        )

        if args.write_video is not None:
            if writer is None:
                args.write_video.parent.mkdir(exist_ok=True, parents=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.write_video), fourcc, 30, (width, height))

            writer.write(draw_overlay(record.image.copy(), record.gt_box, state.box, state.status, state.event))

    if writer is not None:
        writer.release()

    elapsed = max(1e-9, time.perf_counter() - start)
    result = metrics.as_dict()
    result["fps"] = result["frames"] / elapsed
    result["sequence"] = sequence.name
    result["modality"] = args.modality

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
