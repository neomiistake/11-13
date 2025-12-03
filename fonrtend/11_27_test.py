# final_stable_recorder_v5_1.py
# 基於 v5 穩定版架構 (無 crash 風險)
# 新增：CLAHE 影像增強 (專門對抗粉紅畫面與遠距離模糊)
# 調整：YOLO 門檻降至 0.1 (對抗移動模糊)

import cv2
import os
import sys
import numpy as np
import torch
import yaml
from aiortc import RTCSessionDescription, RTCPeerConnection, RTCConfiguration, RTCIceServer
from ultralytics import YOLO
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from collections import defaultdict, deque
import time
import uuid
import requests
import asyncio
import aiohttp
import traceback
import threading
from queue import Queue, Empty
import torch.nn.functional as F
from datetime import datetime

# --- 1. 路徑與配置 ---
try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
    YNET_PROJECT_PATH = os.path.join(PROJECT_ROOT, 'Ynet_kitti_tracking-master')
    if YNET_PROJECT_PATH not in sys.path:
        sys.path.append(YNET_PROJECT_PATH)
    from model import YNet
    from network import modeling
    from utils.image_utils import create_arrow_heatmap, get_patch, create_dist_mat, sampling
except ImportError as e:
    print(f"錯誤：無法導入 Y-Net 相关模组: {e}")
    sys.exit(1)

# --- 配置選項 ---
INPUT_SOURCE_MODE = "WEBRTC"

# --- 通用配置 ---
VIDEO_FILES_DIR = "videos"
RECORD_OUTPUT_DIR = os.path.join(CURRENT_DIR, "web_server", VIDEO_FILES_DIR)
os.makedirs(RECORD_OUTPUT_DIR, exist_ok=True)

FLASK_BACKEND_URL = "http://192.168.196.207:5000"
RECORDING_STATUS_ENDPOINT = f"{FLASK_BACKEND_URL}/recording_status"
UPLOAD_VIDEO_ENDPOINT = f"{FLASK_BACKEND_URL}/upload_recorded_video"
NOTIFY_DANGER_ENDPOINT = f"{FLASK_BACKEND_URL}/notify_danger"

# 建議使用 yolov8m.pt 確保準確度
YOLO_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'yolov8m.pt')
YNET_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'pretrained_models/kitti_ynet_baseline_s8_best.pt')
SEG_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'segmentation_models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')
YNET_CONFIG_PATH = os.path.join(YNET_PROJECT_PATH, r'kitti_train_data/config/kitti.yaml')

SEGMENTATION_INTERVAL = 3
MODEL_INPUT_WIDTH = 320  # Y-Net 需要的寬度
MODEL_INPUT_HEIGHT = 96  # Y-Net 需要的高度
CLASSES_TO_TRACK = [0, 1, 2, 3, 5, 7]
SAVE_VIDEO = True

# --- WebRTC 配置 ---
mediamtx_base_url = "http://192.168.196.73:8889"
stream_paths = ["cam0", "cam1"]
local_camera_indices = [0]

# --- 新增：影像增強函式 (CLAHE) ---
# 這能讓粉紅畫面中的物體輪廓更明顯
def enhance_image(image):
    try:
        # 轉到 LAB 色彩空間，只對 L (亮度) 做增強
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # clipLimit=3.0 表示對比度增強的強度 (預設約 2.0，我們調高一點)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception:
        return image

# --- 寫入影片的執行緒 ---
class VideoWriterThread(threading.Thread):
    def __init__(self, output_path, frame_size, fps=5.0):
        super().__init__()
        self.daemon = True
        self.output_path, self.frame_size, self.fps = output_path, frame_size, fps
        self.write_queue = Queue(maxsize=10)
        self.running = True
        self.writer = None

    def run(self):
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
            if not self.writer.isOpened(): raise IOError("AVC1 failed")
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)

        while self.running or not self.write_queue.empty():
            try:
                frame = self.write_queue.get(timeout=1)
                if self.writer: self.writer.write(frame)
            except Empty: continue
            except Exception: continue
        if self.writer: self.writer.release()

    def add_frame_to_queue(self, frame):
        if self.write_queue.full():
            try: self.write_queue.get_nowait()
            except Empty: pass
        self.write_queue.put_nowait(frame)

    def stop(self): self.running = False

def notify_backend(endpoint, data):
    try:
        requests.post(endpoint, json=data, timeout=2)
    except Exception: pass

# --- WebRTC 接收 ---
def webrtc_receiver_thread(path, frame_queue, shutdown_event):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async def receiver_task():
        while not shutdown_event.is_set():
            pc = None
            try:
                ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
                config = RTCConfiguration(iceServers=ice_servers)
                pc = RTCPeerConnection(configuration=config)

                @pc.on("track")
                async def on_track(track):
                    if track.kind == "video":
                        while not shutdown_event.is_set():
                            try:
                                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                                bgr_frame = frame.to_ndarray(format="bgr24")
                                if frame_queue.full():
                                    try: frame_queue.get_nowait()
                                    except Empty: pass
                                frame_queue.put_nowait(bgr_frame)
                            except asyncio.TimeoutError: break
                            except Exception: break

                url = f"{mediamtx_base_url}/{path}/whep"
                pc.addTransceiver("video", direction="recvonly")
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=pc.localDescription.sdp, headers={"Content-Type": "application/sdp"}, timeout=5) as resp:
                        if resp.status == 201:
                            answer_sdp = await resp.text()
                            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

                while not shutdown_event.is_set():
                    if pc.connectionState in ["failed", "closed"]: break
                    await asyncio.sleep(1)
            except Exception: pass
            finally:
                if pc: await pc.close()
                await asyncio.sleep(2)
    loop.run_until_complete(receiver_task())
    loop.close()

def webcam_receiver_thread(camera_index, frame_queue, shutdown_event):
    cap = cv2.VideoCapture(camera_index)
    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except Empty: pass
        frame_queue.put_nowait(frame)
        time.sleep(0.01)
    cap.release()

# --- 主程式 ---
def main():
    trip_id = f"trip_{uuid.uuid4().hex[:8]}"
    print(f"ID: {trip_id}")
    threading.Thread(target=notify_backend, args=(RECORDING_STATUS_ENDPOINT, {"session_id": trip_id, "status": "start"})).start()

    shutdown_event = threading.Event()
    frame_queues, receiver_threads = {}, []

    if INPUT_SOURCE_MODE == "WEBRTC":
        for path in stream_paths:
            frame_queues[path] = Queue(maxsize=3)
            thread = threading.Thread(target=webrtc_receiver_thread, args=(path, frame_queues[path], shutdown_event), daemon=True)
            receiver_threads.append(thread); thread.start()
    elif INPUT_SOURCE_MODE == "WEBCAM":
        for i, path in zip(local_camera_indices, [f"webcam{i}" for i in local_camera_indices]):
            frame_queues[path] = Queue(maxsize=3)
            thread = threading.Thread(target=webcam_receiver_thread, args=(i, frame_queues[path], shutdown_event), daemon=True)
            receiver_threads.append(thread); thread.start()
    else: return

    print("--- Loading Models ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = YOLO(YOLO_MODEL_PATH)

    with open(YNET_CONFIG_PATH, 'r', encoding='utf-8') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    obs_len = params['obs_len']
    ynet_model = YNet(obs_len=obs_len, pred_len=params['pred_len'], params=params)
    ynet_model.load(YNET_MODEL_PATH)
    ynet_model.model.to(device).eval()

    checkpoint = torch.load(SEG_MODEL_PATH, map_location=device)
    seg_model = modeling.deeplabv3plus_mobilenet(num_classes=19, output_stride=16)
    seg_model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['model_state'].items()})
    seg_model.to(device).eval()

    tracker = ByteTrack()
    input_template = torch.Tensor(create_dist_mat(size=2000)).to(device)

    all_track_histories = {path: defaultdict(lambda: deque(maxlen=obs_len)) for path in frame_queues}
    all_track_predictions = {path: {} for path in frame_queues}
    all_frame_idx = {path: 0 for path in frame_queues}
    all_cached_seg_maps = {path: None for path in frame_queues}
    last_danger_notify_time = {path: 0 for path in frame_queues}
    video_writers = {}

    print("--- Start Monitoring (v5.1 Enhanced) ---")
    time.sleep(2)

    try:
        while not shutdown_event.is_set():
            for path in list(frame_queues.keys()):
                frame_orig = None
                try:
                    while not frame_queues[path].empty():
                        frame_orig = frame_queues[path].get_nowait()
                except Empty: pass

                if frame_orig is None: continue

                height, width, _ = frame_orig.shape

                if path not in video_writers and SAVE_VIDEO:
                    save_path = os.path.join(RECORD_OUTPUT_DIR, f"{path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                    danger_zone = np.array([[int(width*0.25), int(height*0.6)], [int(width*0.75), int(height*0.6)], [int(width*0.95), height-1], [int(width*0.05), height-1]], np.int32)
                    video_writers[path] = {'writer': VideoWriterThread(save_path, (width, height)), 'start_time': time.time(), 'path': save_path, 'danger_zone': danger_zone}
                    video_writers[path]['writer'].start()

                track_histories = all_track_histories[path]
                track_predictions = all_track_predictions[path]
                frame_idx = all_frame_idx[path]
                cached_seg_map = all_cached_seg_maps[path]
                all_frame_idx[path] += 1
                canvas = frame_orig.copy()

                # --- 關鍵修改：使用增強後的圖片進行 YOLO 偵測 ---
                # 這樣不會影響顯示 (canvas) 和追蹤 (frame_orig)，只影響偵測率
                frame_enhanced = enhance_image(frame_orig)

                # 1. YOLO 偵測
                # conf=0.10: 極低門檻，防止移動模糊或距離太遠被過濾
                # 使用 frame_enhanced (增強對比後的圖) 給 AI 看
                results = yolo_model(frame_enhanced, verbose=False, classes=CLASSES_TO_TRACK, conf=0.10, iou=0.5)
                detections_orig = results[0].boxes.data.cpu().numpy()

                # 2. 追蹤 (使用原圖，因為 ByteTrack 習慣原圖的色彩分佈)
                tracked_objects = tracker.update(detections_orig, frame_orig)

                active_track_ids = {int(obj[4]) for obj in tracked_objects}
                track_predictions = {tid: pred for tid, pred in track_predictions.items() if tid in active_track_ids}
                all_track_predictions[path] = track_predictions

                # 3. 分割模型 (縮圖)
                frame_ynet_small = cv2.resize(frame_orig, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
                if frame_idx % SEGMENTATION_INTERVAL == 0 or cached_seg_map is None:
                    img_rgb = cv2.cvtColor(frame_ynet_small, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb.astype(np.float32)/255.0).permute(2,0,1)
                    mean, std = torch.tensor([0.485,0.456,0.406],device=device).view(3,1,1), torch.tensor([0.229,0.224,0.225],device=device).view(3,1,1)
                    with torch.no_grad():
                        seg_logits = seg_model(((img_tensor.to(device) - mean)/std).unsqueeze(0))
                        if isinstance(seg_logits, dict):
                            seg_logits = seg_logits['out']
                        cached_seg_map = torch.argmax(seg_logits.squeeze(), dim=0).cpu().numpy()
                    all_cached_seg_maps[path] = cached_seg_map

                # 4. Y-Net 數據準備 (座標映射)
                tracks_to_predict = []
                scale_x = MODEL_INPUT_WIDTH / width
                scale_y = MODEL_INPUT_HEIGHT / height

                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = obj[:5]
                    track_histories[int(track_id)].append([(x1+x2)/2, y2])
                    if len(track_histories[int(track_id)]) == obs_len:
                        tracks_to_predict.append(int(track_id))

                # 5. Y-Net 預測
                if tracks_to_predict:
                    with torch.no_grad():
                        num_to_predict = len(tracks_to_predict)

                        # 原圖座標 -> Y-Net 座標
                        raw_hist = np.array([list(track_histories[tid]) for tid in tracks_to_predict])
                        batch_hist_ynet = raw_hist.copy()
                        batch_hist_ynet[:, :, 0] *= scale_x
                        batch_hist_ynet[:, :, 1] *= scale_y

                        batch_hist = torch.from_numpy(batch_hist_ynet).float().to(device)
                        vel = batch_hist[:, 1:] - batch_hist[:, :-1]
                        obs_vel = torch.cat([torch.zeros((num_to_predict, 1, 2), device=device), vel], dim=1)
                        acc = obs_vel[:, 1:] - obs_vel[:, :-1]
                        obs_acc = torch.cat([torch.zeros((num_to_predict, 1, 2), device=device), acc], dim=1)

                        h_ynet, w_ynet = MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH
                        seg_map_onehot = F.one_hot(torch.from_numpy(cached_seg_map).long().to(device), 19)
                        seg_map_batch = seg_map_onehot.permute(2,0,1).float().unsqueeze(0).repeat(num_to_predict,1,1,1)

                        vel_map = torch.stack([torch.stack([create_arrow_heatmap(h_ynet,w_ynet,c[0],c[1],v[0],v[1],device=device) for c,v in zip(batch_hist[:,i,:],obs_vel[:,i,:])]) for i in range(obs_len)], dim=1)
                        acc_map = torch.stack([torch.stack([create_arrow_heatmap(h_ynet,w_ynet,c[0],c[1],a[0],a[1],device=device) for c,a in zip(batch_hist[:,i,:],obs_acc[:,i,:])]) for i in range(obs_len)], dim=1)

                        features = ynet_model.model.pred_features(torch.cat([seg_map_batch, vel_map, acc_map], dim=1))
                        pred_waypoint = ynet_model.model.pred_goal(features)[:, params['waypoints']]
                        pred_waypoint_sm = ynet_model.model.sigmoid(pred_waypoint / params['temperature'])
                        goal_samples = sampling(pred_waypoint_sm[:,-1:], num_samples=params.get('num_goals',20)).permute(2,0,1,3)
                        goal_scores = torch.stack([torch.stack([pred_waypoint_sm[i,-1,torch.clamp(g[i,0,1].long(),0,h_ynet-1),torch.clamp(g[i,0,0].long(),0,w_ynet-1)] for i in range(num_to_predict)]) for g in goal_samples])

                        future_samples = []
                        for waypoint in goal_samples:
                            waypoint_map = get_patch(input_template, waypoint.reshape(-1,2).cpu().numpy(), h_ynet, w_ynet).reshape([-1,1,h_ynet,w_ynet])
                            traj_input = [torch.cat([feat, F.interpolate(waypoint_map, size=feat.shape[2:], mode='bilinear', align_corners=False)], dim=1) for feat in features]
                            future_samples.append(ynet_model.model.softargmax(ynet_model.model.pred_traj(traj_input)))

                        future_samples = torch.stack(future_samples)
                        best_indices = torch.argmax(goal_scores, dim=0)
                        best_future = future_samples.permute(1,0,2,3)[torch.arange(num_to_predict), best_indices]

                        for i, track_id in enumerate(tracks_to_predict):
                            all_track_predictions[path][track_id] = best_future[i].cpu().numpy()

                # 6. 繪圖 (反向映射)
                danger_zone_poly = video_writers.get(path, {}).get('danger_zone')
                is_danger = False
                scale_x_inv = width / MODEL_INPUT_WIDTH
                scale_y_inv = height / MODEL_INPUT_HEIGHT

                if danger_zone_poly is not None:
                    for tid, pred_traj in all_track_predictions[path].items():
                        scaled_traj = pred_traj * [scale_x_inv, scale_y_inv]
                        for p in scaled_traj:
                            if cv2.pointPolygonTest(danger_zone_poly, (int(p[0]), int(p[1])), False) >= 0:
                                is_danger = True; break
                        if is_danger: break

                    if is_danger and (time.time() - last_danger_notify_time[path] > 15):
                        last_danger_notify_time[path] = time.time()
                        print(f"[{path}] 危險警報！")
                        threading.Thread(target=notify_backend, args=(NOTIFY_DANGER_ENDPOINT, {
                            "trip_id": trip_id, "event_type": "軌跡預測警告",
                            "description": f"鏡頭 [{path}] 危險！",
                            "timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                        })).start()

                    cv2.polylines(canvas, [danger_zone_poly], True, (0,0,255) if is_danger else (0,255,0), 3)

                for obj in tracked_objects:
                    x1,y1,x2,y2,tid = [int(p) for p in obj[:5]]
                    cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(canvas, f"ID:{tid}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    if tid in track_histories and len(track_histories[tid]) > 1:
                        obs_orig = np.array(track_histories[tid], dtype=np.int32)
                        for k in range(len(obs_orig)-1):
                            cv2.line(canvas, tuple(obs_orig[k]), tuple(obs_orig[k+1]), (255,100,0), 2)

                    if tid in all_track_predictions[path]:
                        pred_orig = (all_track_predictions[path][tid] * [scale_x_inv, scale_y_inv]).astype(int)
                        if tid in track_histories:
                            start_pt = np.array(track_histories[tid][-1], dtype=int)
                            cv2.line(canvas, tuple(start_pt), tuple(pred_orig[0]), (0,0,255), 2)
                        for k in range(len(pred_orig)-1):
                            cv2.line(canvas, tuple(pred_orig[k]), tuple(pred_orig[k+1]), (0,0,255), 2)

                cv2.imshow(f"Monitor - {path}", canvas)
                if path in video_writers: video_writers[path]['writer'].add_frame_to_queue(canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'): shutdown_event.set(); break
    finally:
        print(f"Ending {trip_id}")
        shutdown_event.set()
        for t in receiver_threads: t.join(timeout=1)
        for path, vw in video_writers.items():
            vw['writer'].stop(); vw['writer'].join()
            upload_data = { "trip_id": trip_id, "path": path, "relative_path": os.path.join(VIDEO_FILES_DIR, os.path.basename(vw['path'])).replace("\\","/"), "title": f"Rec-{path}", "date": datetime.now().strftime('%m/%d'), "description": "" }
            threading.Thread(target=notify_backend, args=(UPLOAD_VIDEO_ENDPOINT, upload_data)).start()
        threading.Thread(target=notify_backend, args=(RECORDING_STATUS_ENDPOINT, {"session_id": trip_id, "status": "end"})).start()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()