"""Small, model-independent temporal helpers used by the match pipeline."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class BallObservation:
    point: tuple[float, float]
    source: str
    confidence: float
    candidate: dict | None


class BallTrajectory:
    """Select a plausible ball candidate and bridge only short detection gaps."""

    def __init__(self, max_gap_frames=3, max_speed_fraction_s=0.55):
        self.max_gap_frames = max(0, int(max_gap_frames))
        self.max_speed_fraction_s = float(max_speed_fraction_s)
        self.points = []
        self.missing = 0

    def reset(self):
        self.points.clear()
        self.missing = 0

    def _prediction(self):
        if not self.points:
            return None
        if len(self.points) == 1:
            return np.asarray(self.points[-1], dtype=float)
        velocity = np.asarray(self.points[-1]) - np.asarray(self.points[-2])
        return np.asarray(self.points[-1]) + velocity

    def update(self, candidates, fps, frame_shape):
        fps = max(float(fps), 1.0)
        diagonal = math.hypot(frame_shape[1], frame_shape[0])
        prediction = self._prediction()
        selected = None
        if candidates:
            if prediction is None:
                selected = max(candidates, key=lambda item: item["confidence"])
            else:
                gate = max(24.0, diagonal * self.max_speed_fraction_s / fps * (self.missing + 1))
                plausible = []
                for candidate in candidates:
                    point = np.asarray(candidate["centre"], dtype=float)
                    distance = float(np.linalg.norm(point - prediction))
                    if distance <= gate:
                        score = candidate["confidence"] - 0.35 * distance / gate
                        plausible.append((score, candidate))
                if plausible:
                    selected = max(plausible, key=lambda item: item[0])[1]
        if selected is not None:
            point = tuple(map(float, selected["centre"]))
            self.points.append(point)
            self.points = self.points[-2:]
            self.missing = 0
            return BallObservation(point, "detected", float(selected["confidence"]), selected)
        if prediction is not None and self.missing < self.max_gap_frames:
            self.missing += 1
            point = tuple(map(float, prediction))
            self.points.append(point)
            self.points = self.points[-2:]
            return BallObservation(point, "predicted_gap_fill", 0.0, None)
        self.reset()
        return None
