"""Detection, tracking, and conservative football event heuristics."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import math
import cv2
import numpy as np
from ultralytics import YOLO

from analytics import PitchCalibrator, SiglipTeamClassifier, TacticalRadar
from analytics.team import central_jersey_crop

CLASSES = {"ball", "goalkeeper", "player", "referee"}

@dataclass
class MatchState:
    frames: int = 0
    teams: Counter = field(default_factory=Counter)
    players: set = field(default_factory=set)
    referees: set = field(default_factory=set)
    goalkeepers: set = field(default_factory=set)
    events: list = field(default_factory=list)
    positions: defaultdict = field(default_factory=lambda: defaultdict(list))
    zone_violations: Counter = field(default_factory=Counter)
    possession: Counter = field(default_factory=Counter)

class Tracker:
    def __init__(self, model_path, confidence=.18, tracker_config="Trackers/bytetrack.yaml",
                 image_size=1280, pitch_model=None, pitch_confidence=.50,
                 team_method="colour", team_samples=80, radar=True):
        self.model, self.confidence, self.state = YOLO(model_path), confidence, MatchState()
        self.tracker_config, self.image_size = tracker_config, image_size
        self.last_ball = self.last_owner = self.last_owner_team = None
        self.last_event_frame = -10000
        self.team_colours = {}
        self.track_frames = Counter()
        self.track_team_votes = defaultdict(Counter)
        self.player_positions = defaultdict(deque)
        self.player_pitch_positions = defaultdict(deque)
        self.ball_history = deque(maxlen=3)
        self.owner_streak = 0
        self.pitch = PitchCalibrator(pitch_model, pitch_confidence, image_size) if pitch_model else None
        self.radar = TacticalRadar(self.pitch.config if self.pitch else None)
        self.show_radar = bool(radar and self.pitch)
        self.calibrated_frames = 0
        self.homography_inliers = []
        self.homography_errors = []
        self.team_method = team_method
        self.team_samples = max(10, team_samples)
        self.team_training_crops = []
        self.team_classifier = None
        if team_method == "siglip":
            import torch
            self.team_classifier = SiglipTeamClassifier("cuda" if torch.cuda.is_available() else "cpu")

    def fit_team_classifier(self, video_path):
        """Sample the video before tracking so team labels are stable from frame one."""
        if self.team_classifier is None or self.team_classifier.fitted:
            return
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video for team sampling: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_stride = max(1, total // max(1, self.team_samples // 8))
        crops = []
        frame_no = 0
        while len(crops) < self.team_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % sample_stride == 0:
                result = self.model.predict(frame, imgsz=self.image_size, conf=self.confidence, verbose=False)[0]
                if result.boxes is not None:
                    for xyxy, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.int().tolist()):
                        if result.names[int(cls)] == "player":
                            crop = central_jersey_crop(frame, xyxy)
                            if crop.size:
                                crops.append(crop.copy())
            frame_no += 1
        cap.release()
        if len(crops) < 10:
            raise RuntimeError(f"Only {len(crops)} player crops found; cannot fit SigLIP team clusters.")
        self.team_training_crops = crops[:self.team_samples]
        self.team_classifier.fit(self.team_training_crops)

    def save_embedding_plot(self, path):
        if self.team_classifier is not None and self.team_classifier.fitted:
            self.team_classifier.save_embedding_plot(str(path))

    @staticmethod
    def _centre(box):
        x1, y1, x2, y2 = box
        return ((x1+x2)/2, (y1+y2)/2)

    @staticmethod
    def _jersey_colour(frame, box):
        x1, y1, x2, y2 = map(int, box); h, w = frame.shape[:2]
        crop = frame[max(0,y1):min(h, y1+max(1,(y2-y1)//2)), max(0,x1):min(w,x2)]
        if crop.size == 0: return np.zeros(3)
        # LAB separates the bright white and green kits more robustly than raw BGR.
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        # Remove grass-like pixels before computing a kit colour.
        grass = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90) & (hsv[:, :, 1] > 45)
        pixels = lab[~grass]
        return np.median(pixels, axis=0) if len(pixels) else np.zeros(3)

    def _assign_teams(self, frame, objects):
        people = [o for o in objects if o["class"] == "player"]
        goalkeepers = [o for o in objects if o["class"] == "goalkeeper"]
        if not people:
            return
        labels = None
        crops = [central_jersey_crop(frame, o["box"]) for o in people]
        if self.team_classifier is not None:
            if not self.team_classifier.fitted:
                self.team_training_crops.extend(crop.copy() for crop in crops if crop.size)
                if len(self.team_training_crops) >= self.team_samples:
                    self.team_classifier.fit(self.team_training_crops[:self.team_samples])
            if self.team_classifier.fitted:
                labels = self.team_classifier.predict(crops)
        colours = np.array([self._jersey_colour(frame, o["box"]) for o in people], np.float32)
        if labels is None:
            if len(people) >= 4 and not self.team_colours:
                _, labels, centres = cv2.kmeans(colours, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1), 3, cv2.KMEANS_PP_CENTERS)
                self.team_colours = {0: centres[0], 1: centres[1]}; labels = labels.flatten()
            elif self.team_colours:
                centres = np.array([self.team_colours[0], self.team_colours[1]])
                labels = np.argmin(((colours[:,None]-centres[None])**2).sum(axis=2), axis=1)
            else:
                labels = np.zeros(len(people), dtype=int)
        for obj, label in zip(people, labels):
            candidate = f"team_{int(label)+1}"
            # A track's team is a majority vote across frames, preventing a noisy
            # jersey crop from changing a player's team label mid-video.
            if obj["id"] >= 0:
                votes = self.track_team_votes[obj["id"]]
                votes[candidate] += 1
                obj["team"] = votes.most_common(1)[0][0]
            else:
                obj["team"] = candidate

        # Goalkeeper kits are intentionally excluded from clustering. Associate
        # them with the closest outfield player and stabilize the result by track.
        for goalkeeper in goalkeepers:
            nearest = min(people, key=lambda p: math.dist(self._centre(goalkeeper["box"]), self._centre(p["box"])))
            candidate = nearest.get("team", "unknown")
            if goalkeeper["id"] >= 0:
                votes = self.track_team_votes[goalkeeper["id"]]
                votes[candidate] += 1
                goalkeeper["team"] = votes.most_common(1)[0][0]
            else:
                goalkeeper["team"] = candidate

    @staticmethod
    def _foot(box):
        x1, _, x2, y2 = box
        return ((x1 + x2) / 2, y2)

    def _project_to_pitch(self, frame, objects):
        if self.pitch is None:
            return None
        transformer = self.pitch.infer(frame)
        if transformer is None:
            return None
        anchors, targets = [], []
        detected_balls = [obj for obj in objects if obj["class"] == "ball"]
        primary_ball = max(detected_balls, key=lambda obj: obj["confidence"], default=None)
        for obj in objects:
            if obj["class"] == "ball":
                obj["primary_ball"] = obj is primary_ball
            anchors.append(self._centre(obj["box"]) if obj["class"] == "ball" else self._foot(obj["box"]))
            targets.append(obj)
        if not anchors:
            return transformer
        projected = transformer.transform_points(np.asarray(anchors, dtype=np.float32))
        config = self.pitch.config
        for obj, point in zip(targets, projected):
            x, y = map(float, point)
            if -500 <= x <= config.length + 500 and -500 <= y <= config.width + 500:
                obj["pitch_position"] = (x, y)
                if obj is primary_ball:
                    self.radar.observe_ball((x, y))
        self.calibrated_frames += 1
        self.homography_inliers.append(transformer.inlier_ratio)
        self.homography_errors.append(transformer.reprojection_error)
        return transformer

    def _infer_events(self, objects, frame_no, fps, width, height):
        balls = [o for o in objects if o["class"] == "ball"]
        people = [o for o in objects if o["class"] in {"player", "goalkeeper"}]
        if not balls:
            self.state.possession["unknown"] += 1
            self.last_owner = None
            self.last_owner_team = None
            self.owner_streak = 0
            return
        ball_object = max(balls, key=lambda x:x["confidence"])
        raw_ball = self._centre(ball_object["box"])
        self.ball_history.append(raw_ball)
        ball = tuple(np.median(np.asarray(self.ball_history), axis=0))
        ball_pitch = ball_object.get("pitch_position")
        if ball_pitch is not None:
            candidates = [p for p in people if p.get("pitch_position") is not None]
            owner = min(candidates, key=lambda p: math.dist(ball_pitch, p["pitch_position"]), default=None)
            close_enough = owner is not None and math.dist(ball_pitch, owner["pitch_position"]) < 260
        else:
            owner = min(people, key=lambda p: math.dist(ball, self._foot(p["box"])), default=None)
            player_height = owner["box"][3] - owner["box"][1] if owner else 0
            close_enough = owner is not None and math.dist(ball, self._foot(owner["box"])) < max(25, player_height*.35)
        owner_id = owner.get("id") if close_enough else None
        if owner_id is not None:
            self.state.possession[owner.get("team", "unknown")] += 1
        else:
            self.state.possession["unknown"] += 1
        seconds = frame_no/fps
        if self.last_ball:
            speed = math.dist(ball, self.last_ball)*fps
            if (self.last_owner is not None and owner_id is not None and owner_id != self.last_owner
                    and owner.get("team") == self.last_owner_team
                    and self.owner_streak >= 2 and frame_no-self.last_event_frame > fps*.5):
                self.state.events.append({"time_s":round(seconds,2),"type":"pass","from":self.last_owner,"to":owner_id,"team":owner.get("team","unknown"),"confidence":"heuristic"}); self.last_event_frame=frame_no
            if speed > width*fps*.55 and (ball[0] < width*.12 or ball[0] > width*.88) and frame_no-self.last_event_frame > fps:
                on_target = height*.30 < ball[1] < height*.72
                self.state.events.append({"time_s":round(seconds,2),"type":"shot_on_target" if on_target else "shot_off_target","team":owner.get("team","unknown") if owner else "unknown","confidence":"heuristic"}); self.last_event_frame=frame_no
        self.owner_streak = self.owner_streak + 1 if owner_id == self.last_owner and owner_id is not None else 1
        self.last_ball, self.last_owner = ball, owner_id
        self.last_owner_team = owner.get("team") if owner_id is not None else None

    def analyse_frame(self, frame, frame_no, fps):
        result = self.model.track(frame, persist=True, tracker=self.tracker_config,
                                  imgsz=self.image_size, conf=self.confidence, verbose=False)[0]
        objects=[]; boxes=result.boxes
        if boxes is not None:
            ids = boxes.id.int().tolist() if boxes.id is not None else [-1]*len(boxes)
            for xyxy, cls, conf, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.cls.int().tolist(), boxes.conf.cpu().tolist(), ids):
                label=result.names[int(cls)]
                if label in CLASSES: objects.append({"box":xyxy,"class":label,"confidence":float(conf),"id":int(track_id)})
        self._assign_teams(frame, objects)
        self._project_to_pitch(frame, objects)
        self.state.frames += 1
        for o in objects:
            key=(o["class"], o["id"])
            if o["id"] >= 0:
                self.track_frames[key] += 1
            if o["class"] in {"player","goalkeeper"}:
                team=o.get("team","unknown"); self.state.teams[team]+=1; x,y=self._centre(o["box"])
                if o["class"] == "player" and o["id"] >= 0:
                    # Store each player's own history; pooled team detections cannot
                    # define an individual's normal tactical zone.
                    self.player_positions[o["id"]].append((x/frame.shape[1], y/frame.shape[0]))
                    if o.get("pitch_position") is not None:
                        self.player_pitch_positions[o["id"]].append(o["pitch_position"])
        self._infer_events(objects, frame_no, fps, frame.shape[1], frame.shape[0]); return objects

    def draw(self, frame, objects):
        if self.pitch is not None:
            frame = self.pitch.draw_pitch_overlay(frame, self.pitch.last_transformer)
        colours={"ball":(0,220,255),"goalkeeper":(255,80,255),"referee":(0,165,255),"player":(255,255,255)}
        for o in objects:
            x1,y1,x2,y2=map(int,o["box"]); color=colours[o["class"]]
            if o.get("team")=="team_1": color=(255,120,0)
            if o.get("team")=="team_2": color=(30,50,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            text=f"{o['class']} #{o['id']}"+(f" {o['team']}" if o.get("team") else "")
            cv2.putText(frame,text,(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,2)
        if self.show_radar and self.pitch.last_transformer is not None:
            frame = self.radar.composite(frame, objects)
        return frame

    def report(self, fps):
        counts=defaultdict(Counter)
        for event in self.state.events: counts[event.get("team","unknown")][event["type"]]+=1
        stable = {key for key, frames in self.track_frames.items() if frames >= 10}
        tracked = {label: sum(1 for cls, _ in stable if cls == label)
                   for label in ("player", "goalkeeper", "referee")}
        known = self.state.possession["team_1"] + self.state.possession["team_2"]
        possession_total = known + self.state.possession["unknown"]
        teams={}
        names = sorted((set(self.state.teams) | {"team_1", "team_2"}) - {"unknown"})
        for team in names:
            calibrated = self.calibrated_frames > 0
            source_positions = self.player_pitch_positions if calibrated else self.player_positions
            candidates=[]
            for track_id, points in source_positions.items():
                if ("player", track_id) not in stable or not self.track_team_votes[track_id]:
                    continue
                if self.track_team_votes[track_id].most_common(1)[0][0] == team and len(points) >= 10:
                    candidates.append((self.track_frames[("player", track_id)], np.median(np.asarray(points), axis=0)))
            # Never construct a formation from more than ten outfield players.
            medians=[point for _, point in sorted(candidates, reverse=True, key=lambda item:item[0])[:10]]
            formation="insufficient stable players"
            if len(medians) >= 5:
                points=np.asarray(medians)
                axis=points[:, 0] if calibrated else points[:, 1]
                bins=(0, self.pitch.config.length/4, self.pitch.config.length/2,
                      self.pitch.config.length*3/4, self.pitch.config.length) if calibrated else 4
                bands=np.histogram(axis, bins=bins)[0]
                prefix="pitch-space lines: " if calibrated else "screen-space bands: "
                formation=prefix+"-".join(map(str, map(int, bands)))
            teams[team]={"possession_percent_of_known":round(100*self.state.possession[team]/max(known, 1),1),"formation_estimate":formation,"stable_players_used":len(medians),"events":dict(counts[team])}
        calibration={
            "enabled": self.pitch is not None,
            "coverage_percent": round(100*self.calibrated_frames/max(self.state.frames,1),1),
            "median_inlier_ratio": round(float(np.median(self.homography_inliers)),3) if self.homography_inliers else None,
            "median_reprojection_error_cm": round(float(np.median(self.homography_errors)),1) if self.homography_errors else None,
        }
        limitations=["Player names require jersey-number OCR and a roster.","Fouls and set pieces require a trained action/event model; object detection alone cannot report them reliably.","Passes and shots remain trajectory-based candidates requiring review."]
        if self.pitch is None:
            limitations.append("Pitch calibration is disabled; provide --pitch-model for pitch-space formations, radar and ball territory.")
        team_analysis={"method":self.team_method,"fitted":bool(self.team_classifier and self.team_classifier.fitted),"training_crops":len(self.team_training_crops)}
        return {"frames_analyzed":self.state.frames,"duration_s":round(self.state.frames/fps,2),"tracked_stable":tracked,"possession_coverage_percent":round(100*known/max(possession_total,1),1),"unknown_possession_frames":self.state.possession["unknown"],"team_classification":team_analysis,"pitch_calibration":calibration,"ball_territory":self.radar.report() if self.pitch else None,"teams":teams,"events":self.state.events,"limitations":limitations}
