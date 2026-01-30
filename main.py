import cv2
from ultralytics import YOLO
import argparse

# Load YOLOv8 model (COCO dataset)
model = YOLO("yolov8n.pt")
print("Model YOLOv8 loaded!")

# Counter kategori
count = {
    "botol": 0,
    "bukan_botol": 0
}

# Tracking sederhana
tracks = {}
next_track_id = 0
frame_id = 0

CONF_THRESH = 0.25
IOU_THRESH = 0.4
MAX_AGE = 40
COUNT_DELAY_FRAMES = 5  # berapa frame objek harus terlihat sebelum dihitung

# Fungsi IOU
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denom = boxAArea + boxBArea - interArea
    if denom == 0:
        return 0
    return interArea / denom

# Fungsi kategori mapping (hanya botol vs bukan botol)
def kategori(label):
    l = label.lower()
    if "bottle" in l:
        return "botol"
    else:
        return "bukan_botol"

# Argumen CLI
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Optional path to video file instead of webcam")
args = parser.parse_args()

source = args.video if args.video else 0
cap = cv2.VideoCapture(source)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Kamera/file video tidak dapat dibuka. Periksa indeks kamera atau path file.")
    exit(1)

win_name = "Deteksi Botol"
cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    results = model(frame, verbose=False)[0]

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < CONF_THRESH:
                continue

            cls = int(box.cls.item())
            label = model.names[cls]

            try:
                coords = box.xyxy[0].tolist()
            except Exception:
                coords = [float(v) for v in box.xyxy[0]]

            x1, y1, x2, y2 = map(int, coords)
            detections.append(((x1, y1, x2, y2), label, conf))

    # Tracking & Counting
    for det_bbox, det_label, det_conf in detections:
        matched_id = None
        best_iou = 0

        for tid, t in tracks.items():
            i = iou(det_bbox, t["bbox"])
            if i > best_iou:
                best_iou = i
                matched_id = tid

        if best_iou > IOU_THRESH:
            t = tracks[matched_id]
            t["bbox"] = det_bbox
            t["last_seen"] = frame_id
            t["seen_frames"] = t.get("seen_frames", 1) + 1
        else:
            tracks[next_track_id] = {
                "bbox": det_bbox,
                "label": det_label,
                "last_seen": frame_id,
                "counted": False,
                "seen_frames": 1
            }
            matched_id = next_track_id
            next_track_id += 1

        tcheck = tracks[matched_id]
        if (not tcheck.get("counted", False)) and tcheck.get("seen_frames", 0) >= COUNT_DELAY_FRAMES:
            cat = kategori(tcheck["label"])
            count[cat] += 1
            tcheck["counted"] = True
            print(f"[COUNTED] {cat} => {count[cat]} (label={tcheck['label']})")

        # Draw bounding box
        x1, y1, x2, y2 = det_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{det_label} {det_conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        # Countdown hanya untuk botol
        if not tcheck.get("counted", False):
            cat = kategori(tcheck["label"])
            if cat == "botol":
                remaining = max(0, COUNT_DELAY_FRAMES - tcheck.get("seen_frames", 0))
                cv2.putText(frame, f"Counting in: {remaining}",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2)

    # Cleanup track lama
    for tid in list(tracks.keys()):
        if frame_id - tracks[tid]["last_seen"] > MAX_AGE:
            del tracks[tid]

    # Overlay text
    counter_text = f"Botol: {count['botol']} | Bukan Botol: {count['bukan_botol']}"
    instruction_text = "Tekan Q untuk keluar"

    cv2.rectangle(frame, (5, 5), (900, 60), (0, 0, 0), -1)
    cv2.putText(frame, counter_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)
    cv2.putText(frame, instruction_text, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1)

    cv2.imshow(win_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    print(count)

cap.release()
cv2.destroyAllWindows()