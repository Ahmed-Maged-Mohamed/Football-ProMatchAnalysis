"""Detection, tracking, and explicitly uncertain football analytics."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import math
import cv2
import numpy as np
from ultralytics import YOLO
from analytics import BallTrajectory, PitchCalibrator, SiglipTeamClassifier, TacticalRadar
from analytics.team import central_jersey_crop

CLASSES = {"ball", "goalkeeper", "player", "referee"}

@dataclass
class MatchState:
    frames: int = 0
    teams: Counter = field(default_factory=Counter)
    events: list = field(default_factory=list)
    possession_observed: Counter = field(default_factory=Counter)
    possession_inferred: Counter = field(default_factory=Counter)

class Tracker:
    def __init__(self, model_path, confidence=.10, person_confidence=.18,
                 ball_confidence=.10, tracker_config="Trackers/bytetrack.yaml",
                 image_size=1280, pitch_model=None, pitch_confidence=.50,
                 team_method="colour", team_samples=80, radar=True,
                 ball_gap_frames=3, owner_hold_seconds=1.0):
        self.model = YOLO(model_path)
        self.confidence, self.person_confidence = confidence, person_confidence
        self.ball_confidence = ball_confidence
        self.state = MatchState()
        self.tracker_config, self.image_size = tracker_config, image_size
        self.team_colours, self.track_frames = {}, Counter()
        self.track_team_votes, self.track_classes = defaultdict(Counter), defaultdict(Counter)
        self.player_positions, self.player_pitch_positions = defaultdict(deque), defaultdict(deque)
        self.pitch = PitchCalibrator(pitch_model, pitch_confidence, image_size) if pitch_model else None
        self.radar = TacticalRadar(self.pitch.config if self.pitch else None)
        self.show_radar = bool(radar and self.pitch)
        self.team_method, self.team_samples = team_method, max(10, team_samples)
        self.team_training_crops, self.team_classifier = [], None
        if team_method == "siglip":
            import torch
            self.team_classifier = SiglipTeamClassifier("cuda" if torch.cuda.is_available() else "cpu")
        self.ball_trajectory = BallTrajectory(ball_gap_frames)
        self.ball_diagnostics, self.current_ball = Counter(), None
        self.stable_owner = self.stable_owner_team = None
        self.stable_owner_streak, self.stable_owner_last_frame = 0, -10000
        self.owner_candidate = self.owner_candidate_team = None
        self.owner_candidate_streak = 0
        self.owner_hold_seconds = float(owner_hold_seconds)
        self.last_owner_ball_point = None
        self.detected_pitch_balls = deque(maxlen=5)
        self.last_pass_frame = self.last_shot_frame = -10000

    def fit_team_classifier(self, video_path):
        if self.team_classifier is None or self.team_classifier.fitted:
            return
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise FileNotFoundError(f"Cannot open video: {video_path}")
        total, crops, frame_no = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), [], 0
        sample_stride = max(1, total // max(1, self.team_samples // 8))
        while len(crops) < self.team_samples:
            ok, frame = cap.read()
            if not ok: break
            if frame_no % sample_stride == 0:
                result = self.model.predict(frame, imgsz=self.image_size, conf=self.person_confidence, verbose=False)[0]
                if result.boxes is not None:
                    for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.int().tolist()):
                        if result.names[int(cls)] == "player":
                            crop = central_jersey_crop(frame, box)
                            if crop.size: crops.append(crop.copy())
            frame_no += 1
        cap.release()
        if len(crops) < 10: raise RuntimeError(f"Only {len(crops)} player crops found; cannot fit team clusters.")
        self.team_training_crops = crops[:self.team_samples]
        self.team_classifier.fit(self.team_training_crops)

    def save_embedding_plot(self, path):
        if self.team_classifier is not None and self.team_classifier.fitted:
            self.team_classifier.save_embedding_plot(str(path))

    @staticmethod
    def _centre(box):
        return ((box[0]+box[2])/2, (box[1]+box[3])/2)

    @staticmethod
    def _foot(box):
        return ((box[0]+box[2])/2, box[3])

    @staticmethod
    def _jersey_colour(frame, box):
        x1,y1,x2,y2=map(int,box); h,w=frame.shape[:2]
        crop=frame[max(0,y1):min(h,y1+max(1,(y2-y1)//2)),max(0,x1):min(w,x2)]
        if crop.size == 0: return np.zeros(3)
        hsv, lab = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV), cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)
        grass=(hsv[:,:,0]>=35)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>45)
        pixels=lab[~grass]
        return np.median(pixels,axis=0) if len(pixels) else np.zeros(3)

    def _assign_outfield_teams(self, frame, objects):
        people=[o for o in objects if o["class"]=="player"]
        if not people: return
        labels=None; crops=[central_jersey_crop(frame,o["box"]) for o in people]
        if self.team_classifier is not None:
            if not self.team_classifier.fitted:
                self.team_training_crops.extend(c.copy() for c in crops if c.size)
                if len(self.team_training_crops)>=self.team_samples:
                    self.team_classifier.fit(self.team_training_crops[:self.team_samples])
            if self.team_classifier.fitted and all(c.size for c in crops): labels=self.team_classifier.predict(crops)
        colours=np.array([self._jersey_colour(frame,o["box"]) for o in people],np.float32)
        if labels is None:
            if len(people)>=4 and not self.team_colours:
                _,labels,centres=cv2.kmeans(colours,2,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1),3,cv2.KMEANS_PP_CENTERS)
                self.team_colours={0:centres[0],1:centres[1]}; labels=labels.flatten()
            elif self.team_colours:
                centres=np.array([self.team_colours[0],self.team_colours[1]])
                labels=np.argmin(((colours[:,None]-centres[None])**2).sum(axis=2),axis=1)
            else: labels=np.zeros(len(people),dtype=int)
        for o,label in zip(people,labels):
            candidate=f"team_{int(label)+1}"
            if o["id"]>=0:
                self.track_team_votes[o["id"]][candidate]+=1
                o["team"]=self.track_team_votes[o["id"]].most_common(1)[0][0]
            else: o["team"]=candidate
            o["team_assignment"]="appearance_cluster"

    def _assign_goalkeepers(self, objects):
        by_team=defaultdict(list)
        for p in (o for o in objects if o["class"]=="player"):
            if p.get("team") in {"team_1","team_2"}:
                by_team[p["team"]].append(p.get("pitch_position") or self._centre(p["box"]))
        for keeper in (o for o in objects if o["class"]=="goalkeeper"):
            point=keeper.get("pitch_position") or self._centre(keeper["box"]); scores=[]
            for team in ("team_1","team_2"):
                distances=sorted(math.dist(point,p) for p in by_team[team])
                if len(distances)>=2: scores.append((float(np.median(distances[:3])),team))
            candidate=None
            if len(scores)==2:
                scores.sort()
                if scores[0][0]<=.75*max(scores[1][0],1): candidate=scores[0][1]
            if candidate is None:
                keeper["team"],keeper["team_assignment"]="unknown","withheld_ambiguous"
            else:
                if keeper["id"]>=0:
                    self.track_team_votes[keeper["id"]][candidate]+=1
                    candidate=self.track_team_votes[keeper["id"]].most_common(1)[0][0]
                keeper["team"],keeper["team_assignment"]=candidate,"spatial_cluster_low_confidence"

    def _select_ball(self, objects, fps, frame_shape):
        balls=[o for o in objects if o["class"]=="ball"]
        self.ball_diagnostics["raw_detection_candidates"]+=len(balls)
        for o in balls:
            o["centre"],o["primary_ball"]=self._centre(o["box"]),False
        obs=self.ball_trajectory.update(balls,fps,frame_shape); self.current_ball=None
        if obs is None:
            self.ball_diagnostics["missing_frames"]+=1; return
        if obs.candidate is not None:
            obs.candidate["primary_ball"],obs.candidate["ball_source"]=True,"detected"
            self.current_ball=obs.candidate
            self.ball_diagnostics["accepted_detection_frames"]+=1
            self.ball_diagnostics["rejected_candidates"]+=max(0,len(balls)-1)
        else:
            x,y=obs.point
            virtual={"box":np.asarray((x-4,y-4,x+4,y+4)),"class":"ball","confidence":0.0,"id":-1,
                     "primary_ball":True,"ball_source":"predicted_gap_fill"}
            objects.append(virtual); self.current_ball=virtual
            self.ball_diagnostics["predicted_gap_fill_frames"]+=1
            self.ball_diagnostics["rejected_candidates"]+=len(balls)

    def _project_to_pitch(self, frame, objects):
        if self.pitch is None: return None
        transformer=self.pitch.infer(frame)
        if transformer is None: return None
        if objects:
            anchors=[self._centre(o["box"]) if o["class"]=="ball" else self._foot(o["box"]) for o in objects]
            projected=transformer.transform_points(np.asarray(anchors,np.float32)); config=self.pitch.config
            for o,(x,y) in zip(objects,projected):
                if -500<=x<=config.length+500 and -500<=y<=config.width+500:
                    o["pitch_position"]=(float(x),float(y))
                    o["calibration_source"]="fresh" if self.pitch.last_fresh else "reused"
                    if o.get("primary_ball"):
                        self.radar.observe_ball(o["pitch_position"], o["calibration_source"],
                                                o.get("ball_source", "detected"))
        return transformer

    def _nearest_owner(self, ball, people):
        if ball.get("pitch_position") is not None:
            candidates=[p for p in people if p.get("pitch_position") is not None]
            owner=min(candidates,key=lambda p:math.dist(ball["pitch_position"],p["pitch_position"]),default=None)
            return owner if owner is not None and math.dist(ball["pitch_position"],owner["pitch_position"])<260 else None
        point=self._centre(ball["box"])
        owner=min(people,key=lambda p:math.dist(point,self._foot(p["box"])),default=None)
        height=owner["box"][3]-owner["box"][1] if owner else 0
        return owner if owner is not None and math.dist(point,self._foot(owner["box"]))<max(25,height*.35) else None

    def _maybe_shot(self, ball, frame_no, fps):
        if ball.get("ball_source")!="detected" or ball.get("pitch_position") is None or ball.get("calibration_source")!="fresh": return
        point=np.asarray(ball["pitch_position"],float); self.detected_pitch_balls.append((frame_no,point))
        if len(self.detected_pitch_balls)<2: return
        previous_frame,previous=self.detected_pitch_balls[-2]; elapsed=(frame_no-previous_frame)/max(fps,1)
        if not 0<elapsed<=.5: return
        velocity=(point-previous)/elapsed; speed=float(np.linalg.norm(velocity)); c=self.pitch.config
        directed=(point[0]<c.penalty_box_length and velocity[0]<0) or (point[0]>c.length-c.penalty_box_length and velocity[0]>0)
        if speed>=1200 and directed and frame_no-self.last_shot_frame>fps:
            self.state.events.append({"time_s":round(frame_no/fps,2),"type":"shot_candidate",
                "team":self.stable_owner_team or "unknown","confidence":"low","review_required":True,
                "evidence":{"fresh_calibration":True,"speed_cm_s":round(speed,1),"near_penalty_area":True}})
            self.last_shot_frame=frame_no

    def _infer_events(self, objects, frame_no, fps):
        ball=self.current_ball; people=[o for o in objects if o["class"] in {"player","goalkeeper"}]
        if ball is None:
            self.state.possession_observed["unknown"]+=1
            self.owner_candidate=None; self.owner_candidate_streak=0
            if frame_no-self.stable_owner_last_frame>fps*self.owner_hold_seconds:
                self.stable_owner=self.stable_owner_team=None; self.stable_owner_streak=0
            return
        owner=self._nearest_owner(ball,people)
        owner_id=owner.get("id") if owner is not None and owner.get("id",-1)>=0 else None
        team=owner.get("team","unknown") if owner is not None else "unknown"
        counter=self.state.possession_observed if ball.get("ball_source")=="detected" else self.state.possession_inferred
        counter[team if owner_id is not None else "unknown"]+=1
        if ball.get("ball_source")!="detected": return
        self._maybe_shot(ball,frame_no,fps)
        if owner_id is None:
            self.owner_candidate=None; self.owner_candidate_streak=0; return
        if owner_id==self.owner_candidate: self.owner_candidate_streak+=1
        else: self.owner_candidate,self.owner_candidate_team,self.owner_candidate_streak=owner_id,team,1
        if self.owner_candidate_streak<2: return
        ball_point=ball.get("pitch_position") or self._centre(ball["box"])
        if self.stable_owner==owner_id:
            self.stable_owner_streak+=1; self.stable_owner_last_frame=frame_no; self.last_owner_ball_point=ball_point; return
        gap=frame_no-self.stable_owner_last_frame
        travel=math.dist(ball_point,self.last_owner_ball_point) if self.last_owner_ball_point is not None else 0
        pitch_measure=ball.get("pitch_position") is not None and self.last_owner_ball_point is not None
        if (self.stable_owner is not None and team==self.stable_owner_team and self.stable_owner_streak>=3
                and gap<=fps*self.owner_hold_seconds and travel>=(150 if pitch_measure else 35)
                and frame_no-self.last_pass_frame>fps*.5):
            self.state.events.append({"time_s":round(frame_no/fps,2),"type":"pass_candidate","from":self.stable_owner,
                "to":owner_id,"team":team,"confidence":"low","review_required":True,
                "evidence":{"confirmed_receiver_frames":self.owner_candidate_streak,"gap_frames":int(gap),
                            "travel":round(float(travel),1),"travel_unit":"cm" if pitch_measure else "pixels"}})
            self.last_pass_frame=frame_no
        self.stable_owner,self.stable_owner_team=owner_id,team
        self.stable_owner_streak,self.stable_owner_last_frame=self.owner_candidate_streak,frame_no
        self.last_owner_ball_point=ball_point

    def analyse_frame(self, frame, frame_no, fps):
        result=self.model.track(frame,persist=True,tracker=self.tracker_config,imgsz=self.image_size,conf=self.confidence,verbose=False)[0]
        objects=[]; boxes=result.boxes
        if boxes is not None:
            ids=boxes.id.int().tolist() if boxes.id is not None else [-1]*len(boxes)
            for box,cls,conf,track_id in zip(boxes.xyxy.cpu().numpy(),boxes.cls.int().tolist(),boxes.conf.cpu().tolist(),ids):
                label=result.names[int(cls)]; threshold=self.ball_confidence if label=="ball" else self.person_confidence
                if label in CLASSES and conf>=threshold:
                    objects.append({"box":box,"class":label,"confidence":float(conf),"id":int(track_id)})
        self._select_ball(objects,fps,frame.shape)
        self._assign_outfield_teams(frame,objects); self._project_to_pitch(frame,objects); self._assign_goalkeepers(objects)
        self.state.frames+=1
        for o in objects:
            if o["id"]>=0:
                self.track_frames[(o["class"],o["id"])]+=1; self.track_classes[o["id"]][o["class"]]+=1
            if o["class"] in {"player","goalkeeper"}:
                self.state.teams[o.get("team","unknown")]+=1
                if o["class"]=="player" and o["id"]>=0:
                    x,y=self._centre(o["box"]); self.player_positions[o["id"]].append((x/frame.shape[1],y/frame.shape[0]))
                    if o.get("pitch_position") is not None: self.player_pitch_positions[o["id"]].append(o["pitch_position"])
        self._infer_events(objects,frame_no,fps); return objects

    def draw(self, frame, objects):
        if self.pitch is not None and self.pitch.last_transformer is not None:
            frame=self.pitch.draw_pitch_overlay(frame,self.pitch.last_transformer)
        colours={"ball":(0,220,255),"goalkeeper":(255,80,255),"referee":(0,165,255),"player":(255,255,255)}
        for o in objects:
            if o["class"]=="ball" and not o.get("primary_ball",False): continue
            x1,y1,x2,y2=map(int,o["box"]); color=colours[o["class"]]
            if o.get("team")=="team_1": color=(255,120,0)
            elif o.get("team")=="team_2": color=(30,50,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,1 if o.get("ball_source")=="predicted_gap_fill" else 2)
            if o["class"]=="ball": text="ball" if o.get("ball_source")=="detected" else "ball (predicted)"
            else: text=f"{o['class']} #{o['id']}"+(f" {o['team'][-1]}" if o.get("team") in {"team_1","team_2"} else "")
            cv2.putText(frame,text,(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.40,color,1,cv2.LINE_AA)
        if self.show_radar and self.pitch is not None and self.pitch.last_transformer is not None:
            frame=self.radar.composite(frame,objects)
        return frame

    def report(self, fps):
        stable={k for k,v in self.track_frames.items() if v>=10}; tracking={}
        for label in ("player","goalkeeper","referee"):
            lengths=[v for (cls,_),v in self.track_frames.items() if cls==label]
            tracking[label]={"all_track_ids":len(lengths),"stable_track_ids_ge_10_frames":sum(v>=10 for v in lengths),
                "short_track_ids_lt_10_frames":sum(v<10 for v in lengths),
                "median_track_length_frames":round(float(np.median(lengths)),1) if lengths else None}
        tracking["ids_seen_as_multiple_classes"]=sum(len(c)>1 for c in self.track_classes.values())
        tracking["interpretation"]="Counts diagnose fragmentation/class churn; they do not prove ID switches without labels."
        observed_known=sum(self.state.possession_observed[t] for t in ("team_1","team_2")); observed_total=sum(self.state.possession_observed.values())
        inferred_known=sum(self.state.possession_inferred[t] for t in ("team_1","team_2"))
        event_counts=defaultdict(Counter)
        for event in self.state.events: event_counts[event.get("team","unknown")][event["type"]]+=1
        calibrated=bool(self.pitch and (self.pitch.fresh_frames+self.pitch.reused_frames)); teams={}
        for team in sorted((set(self.state.teams)|{"team_1","team_2"})-{"unknown"}):
            source=self.player_pitch_positions if calibrated else self.player_positions; candidates=[]
            for track_id,points in source.items():
                if (("player",track_id) in stable and self.track_team_votes[track_id]
                        and self.track_team_votes[track_id].most_common(1)[0][0]==team and len(points)>=10):
                    candidates.append((self.track_frames[("player",track_id)],np.median(np.asarray(points),axis=0)))
            medians=[p for _,p in sorted(candidates,reverse=True,key=lambda x:x[0])[:10]]; distribution=None
            if len(medians)>=5:
                axis=np.asarray(medians)[:,0 if calibrated else 1]
                bins=(0,self.pitch.config.length/4,self.pitch.config.length/2,self.pitch.config.length*3/4,self.pitch.config.length) if calibrated else 4
                distribution=list(map(int,np.histogram(axis,bins=bins)[0]))
            teams[team]={"observed_possession_percent_of_known":round(100*self.state.possession_observed[team]/max(observed_known,1),1),
                "outfield_longitudinal_distribution":distribution,"distribution_axis":"absolute_pitch_left_to_right" if calibrated else "screen_top_to_bottom",
                "formation_claimed":False,"stable_players_used":len(medians),"event_candidates":dict(event_counts[team])}
        calibration={"enabled":self.pitch is not None}
        if self.pitch is not None:
            calibration.update({"fresh_frames":self.pitch.fresh_frames,"reused_frames":self.pitch.reused_frames,
                "rejected_or_expired_frames":self.pitch.rejected_frames,"fresh_coverage_percent":round(100*self.pitch.fresh_frames/max(self.state.frames,1),1),
                "usable_including_bounded_reuse_percent":round(100*(self.pitch.fresh_frames+self.pitch.reused_frames)/max(self.state.frames,1),1),
                "max_stale_reuse_frames":self.pitch.max_stale_frames,
                "median_fresh_inlier_ratio":round(float(np.median(self.pitch.fresh_inlier_ratios)),3) if self.pitch.fresh_inlier_ratios else None,
                "median_fresh_reprojection_error_cm":round(float(np.median(self.pitch.fresh_reprojection_errors)),1) if self.pitch.fresh_reprojection_errors else None,
                "quality_note":"Reprojection error is an internal keypoint-fit residual, not verified real-world accuracy."})
        return {"frames_analyzed":self.state.frames,"duration_s":round(self.state.frames/fps,2),
            "thresholds":{"tracker_input":self.confidence,"person":self.person_confidence,"ball":self.ball_confidence},
            "tracking_diagnostics":tracking,"ball_diagnostics":dict(self.ball_diagnostics),
            "possession":{"observed_known_frames":observed_known,"observed_unknown_frames":self.state.possession_observed["unknown"],
                "observed_coverage_percent":round(100*observed_known/max(observed_total,1),1),"inferred_gap_fill_known_frames":inferred_known,
                "inferred_gap_fill_unknown_frames":self.state.possession_inferred["unknown"],"inferred_frames_excluded_from_percentages":True},
            "team_classification":{"method":self.team_method,"fitted":bool(self.team_classifier and self.team_classifier.fitted),"training_crops":len(self.team_training_crops)},
            "pitch_calibration":calibration,"ball_territory":self.radar.report() if self.pitch else None,"teams":teams,"events":self.state.events,
            "limitations":["No ground-truth temporal labels were used, so accuracy and ID-switch improvements are not claimed.",
                "Team labels are unsupervised appearance clusters; ambiguous goalkeeper labels are withheld.",
                "Events are low-confidence review candidates, not measured passes or shots.",
                "Outfield distributions are not formations; territory is not normalized to attack direction.",
                "Player names, fouls, set pieces, and goals require labelled data and dedicated models."]}
