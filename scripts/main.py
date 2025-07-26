import os
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from io_init import init_video_io
from tracker import run_tracking


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 + DeepSORT Video Tracker")
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_path', type=str, default='output/annotated.mp4', help='Output video path')
    parser.add_argument('--save_metadata', action='store_true', help='Dump tracking metadata as JSON/CSV')
    return parser.parse_args()


def load_models():
    person_detector = YOLO("../models/yolo_person_detection.pt")
    box_detector = YOLO("../models/yolo_box_detection.pt") 
    tracker = DeepSort(max_age=30, max_cosine_distance=0.4)
    return person_detector, box_detector, tracker


if __name__ == "__main__":
    args = parse_args()  # from argparse
    person_detector, box_detector, tracker = load_models()
    cap, writer, fps = init_video_io(args.video_path, args.output_path)
    metadata = run_tracking(person_detector, box_detector, tracker, cap, writer, args.save_metadata)
    # if args.save_metadata:
    #     dump_metadata(metadata)