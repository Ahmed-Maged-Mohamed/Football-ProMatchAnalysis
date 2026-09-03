"""Top-down tactical radar and calibrated spatial statistics."""

from collections import Counter

import cv2
import numpy as np

from .pitch import SoccerPitchConfiguration


class TacticalRadar:
    def __init__(self, config=None, scale: float = 0.065, padding: int = 20):
        self.config = config or SoccerPitchConfiguration()
        self.scale = scale
        self.padding = padding
        self.ball_zones = Counter()
        self.calibrated_ball_frames = 0

    def blank_pitch(self) -> np.ndarray:
        width = int(self.config.length * self.scale) + 2 * self.padding
        height = int(self.config.width * self.scale) + 2 * self.padding
        pitch = np.full((height, width, 3), (44, 125, 44), dtype=np.uint8)
        for start, end in self.config.edges:
            p1 = self._pixel(self.config.vertices[start - 1])
            p2 = self._pixel(self.config.vertices[end - 1])
            cv2.line(pitch, p1, p2, (240, 240, 240), 2, cv2.LINE_AA)
        centre = self._pixel((self.config.length / 2, self.config.width / 2))
        cv2.circle(pitch, centre, int(self.config.centre_circle_radius * self.scale), (240, 240, 240), 2)
        return pitch

    def _pixel(self, point) -> tuple[int, int]:
        return (
            int(float(point[0]) * self.scale) + self.padding,
            int(float(point[1]) * self.scale) + self.padding,
        )

    def observe_ball(self, point) -> None:
        if point is None:
            return
        x, y = map(float, point)
        if not (0 <= x <= self.config.length and 0 <= y <= self.config.width):
            return
        longitudinal = ("defensive_third" if x < self.config.length / 3 else
                        "middle_third" if x < 2 * self.config.length / 3 else "attacking_third")
        channel = "left_channel" if y < self.config.width / 3 else "central_channel" if y < 2 * self.config.width / 3 else "right_channel"
        self.ball_zones[longitudinal] += 1
        self.ball_zones[channel] += 1
        self.calibrated_ball_frames += 1

    def draw(self, objects: list[dict]) -> np.ndarray:
        pitch = self.blank_pitch()
        colours = {"team_1": (255, 120, 0), "team_2": (30, 50, 255), "unknown": (210, 210, 210)}
        for obj in objects:
            point = obj.get("pitch_position")
            if point is None:
                continue
            x, y = point
            if not (0 <= x <= self.config.length and 0 <= y <= self.config.width):
                continue
            center = self._pixel(point)
            if obj["class"] == "ball":
                if not obj.get("primary_ball", False):
                    continue
                cv2.circle(pitch, center, 6, (0, 230, 255), -1, cv2.LINE_AA)
            elif obj["class"] == "referee":
                cv2.circle(pitch, center, 6, (0, 165, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(pitch, center, 8, colours.get(obj.get("team"), colours["unknown"]), -1, cv2.LINE_AA)
                cv2.circle(pitch, center, 8, (20, 20, 20), 1, cv2.LINE_AA)
        return pitch

    def composite(self, frame: np.ndarray, objects: list[dict], width_ratio: float = 0.38) -> np.ndarray:
        radar = self.draw(objects)
        target_width = max(240, int(frame.shape[1] * width_ratio))
        ratio = target_width / radar.shape[1]
        radar = cv2.resize(radar, (target_width, int(radar.shape[0] * ratio)))
        x1, y1 = frame.shape[1] - radar.shape[1] - 16, 16
        x2, y2 = x1 + radar.shape[1], y1 + radar.shape[0]
        if x1 < 0 or y2 > frame.shape[0]:
            return frame
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.20, radar, 0.80, 0)
        return frame

    def report(self) -> dict:
        total = self.calibrated_ball_frames
        keys = ("defensive_third", "middle_third", "attacking_third", "left_channel", "central_channel", "right_channel")
        return {
            "calibrated_ball_frames": total,
            "zone_percent": {key: round(100 * self.ball_zones[key] / max(total, 1), 1) for key in keys},
        }
