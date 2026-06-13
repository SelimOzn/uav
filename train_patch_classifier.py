import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pip._internal.models import candidate
from torch.utils.data import DataLoader, TensorDataset

from core.data import AntiUAVSequence
from core.geometry import Box, iou
from core.patch_model import TinyPatchCNN
from core.patch_classifier import crop_patch, select_device

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Patch CNN")
    parser.add_argument("--split-dir", type=Path, default=Path().cwd().parent / "anti_uav_project/Anti-UAV-RGBT/train")
    parser.add_argument("--modality", default="visible", choices=["visible", "infrared"])
    parser.add_argument("--output", type=Path, default=Path("weights/tiny_patch_cnn.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--context_scale", type=float, default=2.0)
    parser.add_argument("--max-sequences", type=int, default=170)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()

def random_box(width: int, height: int, reference: Box | None) -> Box:
    if reference is not None:
        box_w = max(8, int(reference[2] * random.uniform(0.8, 2.0)))
        box_h = max(8, int(reference[3] * random.uniform(0.8, 2.0)))

    else:
        box_w = int(random.uniform(10, 80))
        box_h = int(random.uniform(10, 80))

    x = random.randint(0, max(width - box_w, 0))
    y = random.randint(0, max(height - box_h, 0))

    return float(x), float(y), float(box_w), float(box_h)

def collect_patches(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_dirs = [
        path for path in sorted(args.split_dir.iterdir())
        if path.is_dir() and (path / f"{args.modality}.mp4").exists()
    ][: args.max_sequences]

    patches: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for sequence_dir in sequence_dirs:
        sequence = AntiUAVSequence.from_dir(sequence_dir, args.modality)
        for record in sequence.iter_frames(max_frames=args.max_frames):
            if record.frame_idx % args.frame_stride != 0:
                continue

            height, width = record.image.shape[:2]
            if record.exists and record.gt_box is not None:
                patches.append(crop_patch(record.image, record.gt_box, args.patch_size, args.context_scale))
                labels.append(1.0)

            negatives_added = 0
            tries = 0
            while negatives_added < 2 and tries < 20:
                tries += 1
                candidate = random_box(width, height, record.gt_box)
                if record.gt_box is not None and iou(candidate, record.gt_box) > 0.05:
                    continue
                patches.append(crop_patch(record.image, candidate, args.patch_size, args.context_scale))
                labels.append(0.0)
                negatives_added += 1

    if not patches:
        raise RuntimeError("No training patches collected. Check dataset path and modality.")

    x = np.stack(patches, axis=0).astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    y = np.array(labels, dtype=np.float32)

    return torch.from_numpy(x), torch.from_numpy(y)

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    x, y = collect_patches(args)
    dataset = TensorDataset(x,y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TinyPatchCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)


    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            total_correct += ((probs >= 0.5) == targets).sum().item()
            total_items += targets.numel()
            total_loss += loss.item() * targets.numel()

        print(
            f"epoch={epoch} loss={total_loss / total_items:.4f} "
            f"acc={total_correct / total_items:.4f} samples={total_items}"
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "patch_size": args.patch_size,
                "context_scale": args.context_scale,
            },
            args.output
        )

        print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()