# Pipeline improvement and validation notes

## What changed

- ByteTrack now receives detections at `0.10`, matching `track_low_thresh`; separate post-tracking thresholds retain the `0.18` person filter while allowing ball candidates at `0.10`.
- Ball selection gates candidates against recent motion and labels at most three missing frames as `predicted_gap_fill`. Predictions are reported separately and excluded from possession percentages.
- Possession retains owner context through a short miss. Owner changes need two confirming frames before a low-confidence pass candidate can be emitted.
- The image-edge/pixel-jump shot rule was removed. A shot candidate now needs fresh calibration, pitch-space speed toward a goal, and penalty-area proximity. It is never labelled on/off target.
- Goalkeeper teams are withheld when multi-player spatial clusters do not give a clear answer.
- Homographies expire after eight frames without a valid estimate. Fresh, reused, and rejected/expired frames are separate. Smoothing operates on projected pitch geometry, not matrix coefficients.
- Calibration residual is explicitly an internal fit diagnostic, not verified real-world accuracy.
- Four-bin player histograms are longitudinal distributions, not formations. Ball zones are absolute pitch coordinates, not attack-direction normalized.
- The radar is smaller, secondary ball candidates are hidden, predicted balls are labelled, and player labels are shorter.
- Tracking diagnostics show all/stable/short track counts, median duration, and IDs seen under multiple classes. These do not prove ID switches.

## Reproduce validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

.\.venv\Scripts\python.exe main.py `
  --input input-videos\08fd33_4.mp4 `
  --model models\best-finetune.pt `
  --pitch-model models\pitch-keypoints-best.pt `
  --stride 1 --imgsz 1280 --max-analyzed-frames 220 `
  --output output_videos\improved_smoke.mp4 `
  --report output_videos\improved_smoke_report.json
```

Remove `--max-analyzed-frames` for a full-video run.

## Data needed for measurable model improvement

1. **Ball detector:** label at least 1,000 diverse frames across several matches, including tiny/blurred/occluded balls and hard negatives. Split by match. Measure precision/recall and trajectory gaps on held-out matches.
2. **Pitch keypoints:** label the same 32 landmarks in the exact `SoccerPitchConfiguration.vertices` order. Mark invisible points absent. Use at least 500 varied frames and evaluate keypoint plus hand-checked line/corner error.
3. **Tracking/ReID (optional):** annotate player identities in complete 20–60 second sequences. Report IDF1, HOTA, ID switches, fragments, and mostly-tracked trajectories.
4. **Events (optional):** label temporal boundaries, passer, receiver, team, and outcome, including non-events. Evaluate precision/recall with a timestamp tolerance; heuristic candidates are not ground truth.
5. Provide labelled-data paths and permission to train locally, or explicit cloud budget/authorization. This change performs no upload or training.

Manually audit a fixed test set across the full timeline, not only still frames, and freeze thresholds before comparing versions. More coverage alone is not evidence of more accuracy.
