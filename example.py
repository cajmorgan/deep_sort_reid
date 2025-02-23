import json

import os
from typing import List
import cv2
import numpy as np
from pydantic import TypeAdapter
import torch
from ultralytics import YOLO
from deep_sort_reid.DeepSortReid import DeepSortReid
from deep_sort_reid.types.detection import Detection
from deep_sort_reid.utils.detect_objects import detect_objects_yolo
from deep_sort_reid.utils.extract_features import extract_features_resnet
from deep_sort_reid.utils.misc import get_device
from deep_sort_reid.utils.suppression import non_max_suppression

curr_path = os.getcwd()
file_input_path = curr_path + "/material/walking.mp4"

device = get_device()
model = YOLO("yolo11n.pt")
detections = detect_objects_yolo(file_input_path, model, model_params={
    "stream": True,
    "classes": [0],
    "iou": 0.7,
    "save": False,
    "batch": 200,
    "conf": 0.3,
    "device": device
})


# Save down to cache
detecions_json = []
for frame in detections:
    detecions_i_json = []
    for det in frame:
        detecions_i_json.append(det.model_dump())
    detecions_json.append(detecions_i_json)

with open("detections_cache.json", "w") as f:
    f.write(json.dumps(detecions_json))


with open("detections_cache.json", "r") as f:
    detections_dict = json.loads(f.read())

detections = TypeAdapter(
    List[List[Detection]]).validate_python(detections_dict)

# Apply optional non_max_suppression
detections = non_max_suppression(
    detections, max_overlap=0.7, confidence_dependent=True)

features = extract_features_resnet(
    file_input_path, detections, verbose=True)


features_json = []
for frame in features:
    features_i_json = []
    for feat in frame:
        features_i_json.append(feat.tolist())
    features_json.append(features_i_json)

with open("features_cache.json", "w") as f:
    f.write(json.dumps(features_json))


with open("features_cache.json", "r") as f:
    features_dict = json.loads(f.read())

features = []
for frame_idx, frame in enumerate(features_dict):
    for feat_idx, feat in enumerate(frame):
        detections[frame_idx][feat_idx].feature = torch.Tensor(feat)

deep_sort_reid = DeepSortReid(metric_type="cosine", reid=False, max_since_update=5, reid_similarity_score=0.8,
                              new_track_max_iou=0.5, features_max_distance=0.5, iou_max_distance=0.5)

tracker_results = deep_sort_reid.track(detections)

cap = cv2.VideoCapture(file_input_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_idx = 0
frames = []
while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break

    trackers = tracker_results[frame_idx]
    detection = detections[frame_idx]
    for tracker in trackers:
        x1, y1, x2, y2 = int(tracker.coords.start_x), int(tracker.coords.start_y), int(
            tracker.coords.end_x), int(tracker.coords.end_y)

        cv2.rectangle(frame, (x1, y1), (x2, y2), tracker.color, 4)
        label_height = 40
        cv2.rectangle(frame, (x1, y1 - label_height),
                      (x1 + 120, y1), tracker.color, -1)  # Filled rectangle

        text_color = (255, 255, 255)  # White text for good contrast
        cv2.putText(frame, f'ID: {tracker.track_id}', (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        shadow_offset = 7
        shadow = np.zeros_like(frame)
        cv2.rectangle(shadow, (x1 + shadow_offset, y1 + shadow_offset),
                      (x2 + shadow_offset, y2 + shadow_offset), (50, 50, 50), 4)

    cv2.imshow('Video with Bounding Boxes', frame)

    # Press 'q' to quit the video display
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_idx += 1

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

cap.release()
cv2.destroyAllWindows()
