"""Pitch keypoint inference, validated homography estimation and overlays."""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class SoccerPitchConfiguration:
    """Regulation-sized reference pitch in centimetres (32 landmarks)."""

    width: int = 7000
    length: int = 12000
    penalty_box_width: int = 4100
    penalty_box_length: int = 2015
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100
    edges: tuple = field(default_factory=lambda: (
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
    ))

    @property
    def vertices(self) -> np.ndarray:
        return np.asarray([
            (0, 0),
            (0, (self.width - self.penalty_box_width) / 2),
            (0, (self.width - self.goal_box_width) / 2),
            (0, (self.width + self.goal_box_width) / 2),
            (0, (self.width + self.penalty_box_width) / 2),
            (0, self.width),
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),
            (self.penalty_spot_distance, self.width / 2),
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),
            (self.length / 2, 0),
            (self.length / 2, self.width / 2 - self.centre_circle_radius),
            (self.length / 2, self.width / 2 + self.centre_circle_radius),
            (self.length / 2, self.width),
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),
            (self.length - self.penalty_spot_distance, self.width / 2),
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),
            (self.length, 0),
            (self.length, (self.width - self.penalty_box_width) / 2),
            (self.length, (self.width - self.goal_box_width) / 2),
            (self.length, (self.width + self.goal_box_width) / 2),
            (self.length, (self.width + self.penalty_box_width) / 2),
            (self.length, self.width),
            (self.length / 2 - self.centre_circle_radius, self.width / 2),
            (self.length / 2 + self.centre_circle_radius, self.width / 2),
        ], dtype=np.float32)


class ViewTransformer:
    """RANSAC homography from image pixels to metric pitch coordinates."""

    def __init__(self, source: np.ndarray, target: np.ndarray, ransac_threshold: float = 250.0):
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
            raise ValueError("Source and target must have matching (N, 2) shapes.")
        if len(source) < 4:
            raise ValueError("At least four point pairs are required for homography.")
        self.matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, ransac_threshold)
        if self.matrix is None:
            raise ValueError("Homography matrix could not be calculated.")
        self.matrix = self.matrix / self.matrix[2, 2]
        self.inlier_ratio = float(mask.mean()) if mask is not None else 1.0
        projected = self.transform_points(source)
        self.reprojection_error = float(np.median(np.linalg.norm(projected - target, axis=1)))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        instance = cls.__new__(cls)
        instance.matrix = np.asarray(matrix, dtype=np.float64)
        instance.inlier_ratio = 1.0
        instance.reprojection_error = 0.0
        return instance

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return points.reshape(-1, 2)
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2), self.matrix).reshape(-1, 2)

    def inverse_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        inverse = np.linalg.inv(self.matrix)
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2), inverse).reshape(-1, 2)


class PitchCalibrator:
    """Run a YOLO pose model and maintain a temporally stable homography."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.50,
        image_size: int = 1280,
        min_keypoints: int = 6,
        smoothing: float = 0.35,
        max_stale_frames: int = 8,
        max_reprojection_error: float = 150.0,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.config = SoccerPitchConfiguration()
        self.confidence = confidence
        self.image_size = image_size
        self.min_keypoints = min_keypoints
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.max_stale_frames = max(0, int(max_stale_frames))
        self.max_reprojection_error = float(max_reprojection_error)
        self.last_transformer: Optional[ViewTransformer] = None
        self.last_visible_keypoints = 0
        self.last_fresh = False
        self.stale_age = 0
        self.smoothed_image_vertices: Optional[np.ndarray] = None
        self.fresh_frames = 0
        self.reused_frames = 0
        self.rejected_frames = 0
        self.fresh_inlier_ratios = []
        self.fresh_reprojection_errors = []

    def _reuse_or_expire(self) -> Optional[ViewTransformer]:
        self.last_fresh = False
        if self.last_transformer is None:
            self.rejected_frames += 1
            return None
        self.stale_age += 1
        if self.stale_age > self.max_stale_frames:
            self.last_transformer = None
            self.smoothed_image_vertices = None
            self.rejected_frames += 1
            return None
        self.reused_frames += 1
        return self.last_transformer

    @staticmethod
    def _valid_geometry(transformer: ViewTransformer, frame_shape) -> bool:
        """Reject singular/explosive maps before they enter temporal state."""
        if not np.isfinite(transformer.matrix).all():
            return False
        if np.linalg.cond(transformer.matrix) > 1e8:
            return False
        height, width = frame_shape[:2]
        sample = np.asarray(((0, 0), (width, 0), (width, height), (0, height)), np.float32)
        mapped = transformer.transform_points(sample)
        return bool(np.isfinite(mapped).all() and np.max(np.abs(mapped)) < 1e7)

    def infer(self, frame: np.ndarray) -> Optional[ViewTransformer]:
        result = self.model.predict(frame, imgsz=self.image_size, conf=0.05, verbose=False)[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return self._reuse_or_expire()
        all_xy = result.keypoints.xy.detach().cpu().numpy()
        if result.keypoints.conf is None:
            all_confidence = np.ones(all_xy.shape[:2], dtype=np.float32)
        else:
            all_confidence = result.keypoints.conf.detach().cpu().numpy()
        # Some wide broadcast frames produce overlapping pose proposals. Use
        # the proposal with the largest number of trustworthy landmarks.
        best_index = int(np.argmax((all_confidence >= self.confidence).sum(axis=1)))
        xy = all_xy[best_index]
        confidence = all_confidence[best_index]
        count = min(len(xy), len(self.config.vertices))
        valid = (confidence[:count] >= self.confidence) & np.isfinite(xy[:count]).all(axis=1)
        self.last_visible_keypoints = int(valid.sum())
        if self.last_visible_keypoints < self.min_keypoints:
            return self._reuse_or_expire()
        try:
            estimate = ViewTransformer(xy[:count][valid], self.config.vertices[:count][valid])
        except ValueError:
            return self._reuse_or_expire()
        if (estimate.inlier_ratio < 0.60
                or estimate.reprojection_error > self.max_reprojection_error
                or not self._valid_geometry(estimate, frame.shape)):
            return self._reuse_or_expire()

        # Smooth image-space pitch geometry, then solve a new homography. Direct
        # coefficient-wise homography averaging has no projective meaning.
        image_vertices = estimate.inverse_points(self.config.vertices)
        if not np.isfinite(image_vertices).all():
            return self._reuse_or_expire()
        if self.smoothed_image_vertices is None:
            self.smoothed_image_vertices = image_vertices
        else:
            a = self.smoothing
            self.smoothed_image_vertices = a * image_vertices + (1.0 - a) * self.smoothed_image_vertices
        try:
            smoothed = ViewTransformer(self.smoothed_image_vertices, self.config.vertices)
        except ValueError:
            return self._reuse_or_expire()
        smoothed.inlier_ratio = estimate.inlier_ratio
        smoothed.reprojection_error = estimate.reprojection_error
        self.last_transformer = smoothed
        self.last_fresh = True
        self.stale_age = 0
        self.fresh_frames += 1
        self.fresh_inlier_ratios.append(estimate.inlier_ratio)
        self.fresh_reprojection_errors.append(estimate.reprojection_error)
        return self.last_transformer

    def draw_pitch_overlay(self, frame: np.ndarray, transformer: Optional[ViewTransformer]) -> np.ndarray:
        if transformer is None:
            return frame
        vertices = transformer.inverse_points(self.config.vertices)
        overlay = frame.copy()
        for start, end in self.config.edges:
            p1, p2 = vertices[start - 1], vertices[end - 1]
            if np.isfinite([*p1, *p2]).all():
                cv2.line(overlay, tuple(np.int32(p1)), tuple(np.int32(p2)), (80, 255, 80), 2, cv2.LINE_AA)
        return cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
