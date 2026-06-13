import numpy as np

class KalmanBoxTracker:
    """Constant-velocity Kalman filter for a single image-space box."""

    def __init__(self, process_noise: float = 8.0, measurement_noise: float = 20.0):
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.p = np.eye(6, dtype=np.float32) * 1000.0
        self.f = np.eye(6, dtype=np.float32)
        self.h = np.zeros((4, 6), dtype=np.float32)
        self.h[0, 0] = 1.0
        self.h[1, 1] = 1.0
        self.h[2, 4] = 1.0
        self.h[3, 5] = 1.0
        self.q = np.eye(6, dtype=np.float32) * process_noise
        self.r = np.eye(4, dtype=np.float32) * measurement_noise
        self.initialized = False

    def initiate(self, box: tuple[float, float, float, float]) -> None:
        cx = box[0] + box[2] * 0.5
        cy = box[1] + box[3] * 0.5
        self.x[:, 0] = [cx, cy, 0.0, 0.0, box[2], box[3]]
        self.p = np.eye(6, dtype=np.float32) * 10.0
        self.initialized = True

    def predict(self) -> tuple[float, float, float, float] | None:
        if not self.initialized:
            return None

        self.f[:] = np.eye(6, dtype=np.float32)
        self.f[0, 2] = 1.0
        self.f[1, 3] = 1.0
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        return self.current_box()

    def update(self, box: tuple[float, float, float, float]) -> None:
        if not self.initialized:
            self.initiate(box)
            return

        z = np.array(
            [[box[0] + box[2] * 0.5], [box[1] + box[3] * 0.5], [box[2]], [box[3]]],
            dtype=np.float32,
        )
        y = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.p = (np.eye(6, dtype=np.float32) - k @ self.h) @ self.p

    def current_box(self) -> tuple[float, float, float, float]:
        cx, cy, _, _, w, h = self.x[:, 0].tolist()
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        return float(cx - w * 0.5), float(cy - h * 0.5), w, h

    def uncertainty(self) -> float:
        return float(np.trace(self.p[:2, :2]))

    def box_from_state(self, state) -> tuple[float, float, float, float]:
        cx, cy, _, _, w, h = state[:, 0].tolist()
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        return float(cx - w * 0.5), float(cy - h * 0.5), w, h

    def predicted_box(self):
        if not self.initialized:
            return None

        f = self.f.copy()
        f[:] = np.eye(6, dtype=np.float32)
        f[0, 2] = 1.0
        f[1, 3] = 1.0
        x_pred = f @ self.x
        return self.box_from_state(x_pred)

