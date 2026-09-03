import argparse
import json
from pathlib import Path
import cv2
from Trackers import Tracker

def main():
    parser = argparse.ArgumentParser(description="Analyse a football video with the supplied YOLO model.")
    parser.add_argument("--input", default="input-videos/08fd33_4.mp4")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--output", default="output_videos/annotated_match.mp4")
    parser.add_argument("--report", default="output_videos/match_report.json")
    parser.add_argument("--stride", type=int, default=2, help="Analyse every Nth frame (1 = all frames).")
    parser.add_argument("--imgsz", type=int, default=1280, help="Detector inference resolution; higher improves small-ball detection.")
    parser.add_argument("--tracker", default="Trackers/bytetrack.yaml", help="Ultralytics tracker configuration file.")
    parser.add_argument("--pitch-model", default="models/pitch-keypoints-best.pt",
                        help="YOLO pose checkpoint trained on the 32 pitch landmarks; pass an empty string to disable.")
    parser.add_argument("--pitch-conf", type=float, default=.50, help="Minimum pitch-landmark confidence used for homography.")
    parser.add_argument("--team-method", choices=("colour", "siglip"), default="colour",
                        help="Team clustering backend. SigLIP requires requirements-advanced.txt.")
    parser.add_argument("--team-samples", type=int, default=80, help="Player crops used to fit SigLIP/UMAP/K-Means.")
    parser.add_argument("--embedding-plot", default="output_videos/team_embeddings.png",
                        help="SigLIP/UMAP diagnostic plot path.")
    parser.add_argument("--no-radar", action="store_true", help="Disable the top-down radar overlay.")
    args = parser.parse_args()
    if args.stride < 1: parser.error("--stride must be at least 1")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps/args.stride, (width,height))
    tracker, frame_no = Tracker(
        args.model,
        tracker_config=args.tracker,
        image_size=args.imgsz,
        pitch_model=args.pitch_model,
        pitch_confidence=args.pitch_conf,
        team_method=args.team_method,
        team_samples=args.team_samples,
        radar=not args.no_radar,
    ), 0
    if args.team_method == "siglip":
        print("Collecting player crops and fitting SigLIP/UMAP team clusters...", flush=True)
        tracker.fit_team_classifier(args.input)
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_no % args.stride == 0:
            objects = tracker.analyse_frame(frame, frame_no//args.stride, fps/args.stride)
            writer.write(tracker.draw(frame, objects))
        if frame_no and frame_no % (100 * args.stride) == 0:
            percent = (100 * frame_no / total_frames) if total_frames else 0
            print(f"Analysed frame {frame_no}/{total_frames} ({percent:.1f}%)", flush=True)
        frame_no += 1
    cap.release(); writer.release()
    if args.team_method == "siglip":
        Path(args.embedding_plot).parent.mkdir(parents=True, exist_ok=True)
        tracker.save_embedding_plot(args.embedding_plot)
    with open(args.report, "w", encoding="utf-8") as report:
        json.dump(tracker.report(fps/args.stride), report, indent=2)
    print(f"Wrote {args.output} and {args.report}")

if __name__ == "__main__": main()
