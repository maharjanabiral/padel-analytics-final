import json
import cv2
import numpy as np
import torch
from collections import deque, defaultdict
from ultralytics import YOLO
from model import PadelGRU


input_video = "data/input_sample_video.mp4"
result_json = "outputs/shot_results.json"
seq_len = 20
input_size = 54
hidden_size = 16
num_layers = 2
num_classes = 4
classes = ["backhand", "forehand", "ready_position", "serve"]
confidence_threshold = 0.5
classify_every_n = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

_inv_w = 1.0 / 1280.0
_inv_h = 1.0 / 720.0

PLAYER_COLORS = [
    (255, 100,  50),
    ( 50, 180, 255),
    (180,  50, 255),
    ( 50, 220, 100),
]
BALL_COLOR   = (0, 255, 255)
RACKET_COLOR = (0, 255,   0)

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

pose_model = YOLO("yolov8s-pose.pt").to(device)
det_model  = YOLO("yolov8x.pt").to(device)

gru_model = PadelGRU(input_size, hidden_size, num_layers, num_classes).to(device)
gru_model.load_state_dict(torch.load("models/best_tennis_lstm.pth", map_location=device))
gru_model.eval()

player_buffers   : dict[int, deque] = defaultdict(lambda: deque(maxlen=seq_len))
player_labels    : dict[int, str]   = {}
player_confs     : dict[int, float] = {}
player_color_idx : dict[int, int]   = {}
_next_color_idx = 0

shot_results: list[dict] = []

def get_player_color(track_id):
    global _next_color_idx
    if track_id not in player_color_idx:
        player_color_idx[track_id] = _next_color_idx % len(PLAYER_COLORS)
        _next_color_idx += 1
    return PLAYER_COLORS[player_color_idx[track_id]]


def preprocess_keypoints(kp):
    kp = kp.copy().astype(np.float32)
    kp[kp[:, 2] == 0, :2] = 0.0
    hip_center = (kp[11, :2] + kp[12, :2]) * 0.5
    kp[:, :2] -= hip_center
    kp[:, 0] *= _inv_h
    kp[:, 1] *= _inv_w
    return kp.flatten()


@torch.no_grad()
def classify_batch(buffers):
    ready_ids = [tid for tid, buf in buffers.items() if len(buf) == seq_len]
    if not ready_ids:
        return {}
    x = torch.tensor(
        np.stack([np.array(buffers[tid]) for tid in ready_ids]),
        dtype=torch.float32,
    ).to(device)
    probs = torch.softmax(gru_model(x), dim=1)
    confs, idxs = probs.max(dim=1)
    return {ready_ids[i]: (classes[idxs[i].item()], confs[i].item())
            for i in range(len(ready_ids))}


def draw_skeleton(frame, kps_xy, color):
    for x, y in kps_xy:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
    for i, j in SKELETON:
        x1, y1 = kps_xy[i]; x2, y2 = kps_xy[j]
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)


def draw_player_label(frame, track_id, bbox_xyxy, label, conf):
    color = get_player_color(track_id)
    x1, y1 = int(bbox_xyxy[0]), int(bbox_xyxy[1])
    text = f"P{track_id} | {label.upper()}  {conf * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    ty = max(y1 - 10, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (20, 20, 20), -1)
    cv2.putText(frame, text, (x1 + 3, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_objects(frame, det_results):
    if det_results[0].boxes is None:
        return
    for box in det_results[0].boxes:
        cls_id = int(box.cls[0])
        name   = det_model.names[cls_id]
        conf   = float(box.conf[0])
        if name not in ("sports ball", "tennis racket"):
            continue

        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        color = BALL_COLOR if name == "sports ball" else RACKET_COLOR
        display_name = "BALL" if name == "sports ball" else "RACKET"

        if name == "sports ball":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 6, color, -1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, f"{display_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
def draw_hud(frame, shot_results):
    counts = {"backhand": 0, "forehand": 0, "serve": 0}
    box_w = 220
    box_h = 90
    for s in shot_results:
            if s["shot_type"] in counts:
                counts[s["shot_type"]] += 1
 
    
    hud_x = frame.shape[1] - box_w - 20
    hud_y = frame.shape[0] - box_h - 60
    cv2.rectangle(frame, (hud_x + box_w, hud_y + box_h),
                      (frame.shape[1] - 10, hud_y + 90), (20, 20, 20), -1)
    cv2.putText(frame, "SHOT COUNTS", (hud_x, hud_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    for idx, (shot, count) in enumerate(counts.items()):
        cv2.putText(frame, f"{shot:<12}: {count}",
            (hud_x, hud_y + 38 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)



def main():
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {input_video}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose_result = pose_model.track(frame, persist=True, verbose=False)[0]
        det_results = det_model(frame, verbose=False)

        if (pose_result.keypoints is not None
                and pose_result.boxes is not None
                and pose_result.boxes.id is not None):

            track_ids = pose_result.boxes.id.int().tolist()
            kps_data  = pose_result.keypoints.data
            kps_xy    = pose_result.keypoints.xy.cpu().numpy()
            boxes     = pose_result.boxes.xyxy

            for i, track_id in enumerate(track_ids):
                kp17 = kps_data[i].cpu().numpy()
                neck = np.array([
                    (kp17[5,0] + kp17[6,0]) / 2,
                    (kp17[5,1] + kp17[6,1]) / 2,
                    min(kp17[5,2], kp17[6,2]),
                ], dtype=np.float32)
                player_buffers[track_id].append(
                    preprocess_keypoints(np.vstack([kp17, neck]))
                )

            if frame_count % classify_every_n== 0:
                for tid, (lbl, conf) in classify_batch(player_buffers).items():
                    if conf >= confidence_threshold:
                        if player_labels.get(tid) != lbl:
                            shot_results.append({
                                "player_id": tid,
                                "shot_type": lbl,
                                "frame":     frame_count,
                            })
                        player_labels[tid] = lbl
                        player_confs[tid]  = conf

            for i, track_id in enumerate(track_ids):
                color = get_player_color(track_id)
                draw_skeleton(frame, kps_xy[i], color)
                if (track_id in player_labels
                        and player_confs.get(track_id, 0) >= confidence_threshold):
                    draw_player_label(frame, track_id,
                                      boxes[i].cpu().numpy(),
                                      player_labels[track_id],
                                      player_confs[track_id])

        draw_objects(frame, det_results)
        draw_hud(frame, shot_results)

       
        cv2.imshow("Padel Analytics", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(result_json, "w") as f:
        json.dump(shot_results, f, indent=2)
    print(f"Saved {len(shot_results)} shot events → {result_json}")


if __name__ == "__main__":
    main()