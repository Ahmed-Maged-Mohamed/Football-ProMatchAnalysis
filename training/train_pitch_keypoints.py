"""Fine-tune YOLOv8x-pose for the 32 football pitch landmarks."""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Pose dataset data.yaml")
    parser.add_argument("--model", default="yolov8x-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/pitch-keypoints")
    args = parser.parse_args()
    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name="yolov8x-pose-football-pitch",
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        plots=True,
        save_period=5,
    )


if __name__ == "__main__":
    main()
