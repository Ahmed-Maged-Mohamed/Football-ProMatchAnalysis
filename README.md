# Football Pro Match Analysis

Run the analyser with:

```powershell
.\.venv\Scripts\python.exe main.py --input input-videos/08fd33_4.mp4
```

It writes an annotated video and `output_videos/match_report.json`. Use
`--stride 1` for maximum temporal detail, or a larger stride for faster runs.
For the bundled model, the default `--stride 2 --imgsz 1280` is a good
quality/speed trade-off.

The bundled detector identifies ball, player, goalkeeper, and referee. The
pipeline tracks those objects, clusters kit colours into two teams, estimates
screen-space formation bands from stable player tracks, and emits conservative
pass and shot candidates. Every inferred event is marked `heuristic` in the JSON.
Unknown ball-owner frames are reported separately rather than being assigned to
one team.

## Advanced calibrated analysis

The project now also supports the pipeline demonstrated in Roboflow's Football
AI tutorials:

- SigLIP visual embeddings, UMAP and K-Means for team clustering.
- YOLO pose inference for 32 pitch landmarks.
- RANSAC homography and perspective transformation.
- Virtual pitch-line overlays on the broadcast frame.
- A top-down tactical radar showing players, referees and the ball.
- Calibrated ball territory statistics by pitch third and lateral channel.

Train a pitch-landmark model from a YOLO pose dataset:

```powershell
.\.venv\Scripts\python.exe training\train_pitch_keypoints.py `
  --data training\football-pitch-keypoints\data.yaml `
  --model yolov8x-pose.pt
```

Install the optional embedding dependencies and run all advanced features:

```powershell
.\.venv\Scripts\pip.exe install -r requirements-advanced.txt

.\.venv\Scripts\python.exe main.py `
  --input input-videos\08fd33_4.mp4 `
  --model models\best-finetune.pt `
  --pitch-model models\pitch-keypoints-best.pt `
  --team-method siglip `
  --stride 1 `
  --imgsz 1280 `
  --output output_videos\advanced_match.mp4 `
  --report output_videos\advanced_match_report.json
```

`--pitch-model` is deliberately optional: without it, detection and tracking
still run, but the report clearly marks calibration as disabled and withholds
pitch-space territory. The pose model must use the same 32-keypoint order as
`SoccerPitchConfiguration.vertices` in `analytics/pitch.py`.

## What requires further training

Player names need jersey-number OCR plus a match roster. Reliable foul,
free-kick/corner/penalty, and goal classification needs labelled video clips and
a temporal action-recognition model. Formation and territory results should only
be treated as tactical measurements when pitch-calibration coverage and homography
quality in the JSON report are high.
