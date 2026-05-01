import json
import cv2
from pathlib import Path
from collections import defaultdict

video_name = "uid_vid_00000"
model_name = "yolo11n"

base = Path("F:/Alale/YOlo v11/personpath_yolo11")

video_path = base / "tracking-dataset/dataset/personpath22/raw_data/videos" / f"{video_name}.mp4"
pred_path  = base / f"outputs/detections/{model_name}/{video_name}_preds.json"
out_dir = base / "outputs/visuals"
out_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
preds_raw = json.load(open(pred_path))

print("video exists:", video_path.exists())
print("video opened:", cap.isOpened())
print("pred exists:", pred_path.exists())
print("prediction type:", type(preds_raw))

preds_by_frame = defaultdict(list)

if isinstance(preds_raw, list):
    for det in preds_raw:
        frame_id = det.get("frame_id", det.get("frame", det.get("image_id")))
        if frame_id is None:
            continue
        preds_by_frame[int(frame_id)].append(det)

elif isinstance(preds_raw, dict):
    for k, v in preds_raw.items():
        if isinstance(v, list):
            preds_by_frame[int(k)].extend(v)

print("frames with predictions:", len(preds_by_frame))

frame_id = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    dets = preds_by_frame.get(frame_id, [])

    for det in dets:
        if "bbox" in det:
            x, y, w, h = det["bbox"]
        else:
            x = det.get("x1", det.get("x", 0))
            y = det.get("y1", det.get("y", 0))
            if "x2" in det and "y2" in det:
                w = det["x2"] - x
                h = det["y2"] - y
            else:
                w = det.get("w", 0)
                h = det.get("h", 0)

        x, y, w, h = map(int, [x, y, w, h])
        if w <= 0 or h <= 0:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        score = det.get("score", det.get("confidence", None))
        if score is not None:
            cv2.putText(frame, f"{score:.2f}", (x, max(20, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if frame_id % 50 == 0:
        cv2.imwrite(str(out_dir / f"{video_name}_{frame_id}.jpg"), frame)
        saved += 1

    frame_id += 1

cap.release()
print(f"Done. Saved {saved} images in outputs/visuals/")