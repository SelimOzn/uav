
from pathlib import Path

import cv2
import numpy as np
import torch

from core.geometry import Box, clip_box, expand_box
from core.motion import Detection
from core.patch_model import TinyPatchCNN

def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

class PatchClassifier:
    def __init__(self,
                 checkpoint_path: str | Path,
                 device: str = "auto",
                 patch_size: int = 64,
                 context_scale: float = 2.0,
                 threshold: float = 0.45,
                 batch_size: int = 64):
        self.device = select_device(device)
        self.patch_size = patch_size
        self.context_scale = context_scale
        self.threshold = threshold
        self.batch_size = batch_size

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "patch_size" in checkpoint:
            self.patch_size = int(checkpoint["patch_size"])

        self.model = TinyPatchCNN().to(self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def filter(self, frame: np.ndarray, detections: list[Detection], top_k: int | None = None) -> list[Detection]:
        if not detections:
            return []

        kept: list[Detection] = []
        for start in range(0, len(detections), self.batch_size):
            batch = detections[start:start + self.batch_size]
            tensor = self._batch_to_tensor(frame, [det.box for det in batch]).to(self.device)
            probs = torch.sigmoid(self.model(tensor)).detach().cpu().numpy().tolist()
            for detection, prob in zip(batch, probs):
                if prob >= self.threshold:
                    kept.append(Detection(box=detection.box, score=float(prob), source="motion+patch_cnn"))

        kept.sort(key=lambda x: x.score, reverse=True)
        return kept[:top_k] if top_k is not None else kept

    def _batch_to_tensor(self, frame: np.ndarray, boxes: list[Box]) -> torch.Tensor:
        patches = [crop_patch(frame, box, self.patch_size, self.context_scale) for box in boxes]
        array = np.stack(patches, axis=0).astype(np.float32) / 255.0
        array = np.transpose(array, (0, 3, 1, 2))
        return torch.from_numpy(array)


def crop_patch(frame: np.ndarray, box: Box, patch_size: int, context_scale: float = 2.0) -> np.ndarray:
    height, width = frame.shape[:2]
    crop_box = expand_box(box, context_scale, width, height)
    x, y, w, h = clip_box(crop_box, width, height)
    crop = frame[int(y):int(y + h), int(x):int(x + w)]
    if crop.size == 0:
        crop = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
