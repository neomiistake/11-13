# final_stable_recorder.py (修正版：恢復您的原始影像邏輯 + 加入儀表板連動)

import cv2
import os
import sys
import numpy as np
import torch
import yaml
from aiortc import RTCSessionDescription, RTCPeerConnection
from ultralytics import YOLO
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from collections import defaultdict, deque
import time
import uuid
import requests
import asyncio
import aiohttp
from aiortc import RTCConfiguration, RTCIceServer
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
    print(f"錯誤：無法導入 Y-Net 相關模組: {e}")
    sys.exit(1)

# --- 配置選項 ---
# 【請確認這裡】如果您是用樹莓派推流，請選 WEBRTC
INPUT_SOURCE_MODE = "WEBCAM"

# --- 通用配置 ---
VIDEO_FILES_DIR = "videos"
RECORD_OUTPUT_DIR = os.path.join(CURRENT_DIR, "web_server", VIDEO_FILES_DIR)
os.makedirs(RECORD_OUTPUT_DIR, exist_ok=True)
print(f"影片將儲存至: {RECORD_OUTPUT_DIR}")

# 【儀表板連線設定】(修改為本機 IP，確保能連上 app.py)
FLASK_BACKEND_URL = "http://192.168.196.207:5000"
#張:"http://192.168.196.230:5000"
#李:"http://192.168.196.207:5000"

RECORDING_STATUS_ENDPOINT = f"{FLASK_BACKEND_URL}/recording_status"
UPLOAD_VIDEO_ENDPOINT = f"{FLASK_BACKEND_URL}/upload_recorded_video"
NOTIFY_DANGER_ENDPOINT = f"{FLASK_BACKEND_URL}/notify_danger"

YOLO_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'yolov8n.pt')
YNET_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'pretrained_models/kitti_ynet_baseline_s8_best.pt')
SEG_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'segmentation_models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')
YNET_CONFIG_PATH = os.path.join(YNET_PROJECT_PATH, r'kitti_train_data/config/kitti.yaml')

SEGMENTATION_INTERVAL = 3
MODEL_INPUT_WIDTH = 320
MODEL_INPUT_HEIGHT = 96
CLASSES_TO_TRACK = [0, 1, 2, 3, 5, 7]
SAVE_VIDEO = True

# --- WebRTC 配置 (保留您原本的 IP) ---
mediamtx_base_url = "http://10.218.242.149:8889"
stream_paths = ["cam0", "cam1"]

# --- 本地 Webcam 配置 ---
local_camera_indices = [0]


# --- 影片寫入執行緒 ---
class VideoWriterThread(threading.Thread):
    def __init__(self, output_path, frame_size, fps=5.0):
        super().__init__()
        self.daemon = True
        self.output_path, self.frame_size, self.fps = output_path, frame_size, fps
        self.write_queue = Queue(maxsize=120)
        self.running = True
        self.writer = None

    def run(self):
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
            if not self.writer.isOpened(): raise IOError("AVC1 failed")
            print(f"影片寫入執行緒(avc1)已啟動: {os.path.basename(self.output_path)}")
        except Exception:
            print(f"警告: AVC1 編碼器不可用, 降級至 MP4V")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
        while self.running or not self.write_queue.empty():
            try:
                frame = self.write_queue.get(timeout=1)
                if self.writer: self.writer.write(frame)
            except Empty:
                continue
        if self.writer: self.writer.release()
        print(f"影片 {os.path.basename(self.output_path)} 寫入完成。")

    def add_frame_to_queue(self, frame):
        if not self.write_queue.full(): self.write_queue.put_nowait(frame)

    def stop(self):
        self.running = False


def notify_backend(endpoint, data):
    try:
        # 設定短一點的 timeout 避免卡住錄影
        response = requests.post(endpoint, json=data, timeout=2)
        response.raise_for_status()
        if "recording_status" in endpoint:
            print(f"✅ 成功通知後端: {data.get('status')}")
        return response.json()
    except requests.exceptions.RequestException as e:
        # print(f"通知後端失敗: {e}") # 為了畫面乾淨，可以註解掉這行
        return None


# --- WebRTC 接收執行緒 (完全恢復您原本的邏輯) ---
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
                pc.RTCP_REPORTS_DEFAULT = True
                first_frame_received = asyncio.Queue(maxsize=1)

                @pc.on("track")
                async def on_track(track):
                    print(f"[{path}] 接收到 Track: {track.kind}")
                    if track.kind == "video":
                        try:
                            first_frame = await asyncio.wait_for(track.recv(), timeout=15.0)
                            print(f"[{path}] ✅ 成功接收到第一個有效影像幀！")
                            if not frame_queue.full():
                                frame_queue.put_nowait(first_frame.to_ndarray(format="bgr24"))
                            await first_frame_received.put(True)
                            while not shutdown_event.is_set():
                                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                                if not frame_queue.full():
                                    frame_queue.put_nowait(frame.to_ndarray(format="bgr24"))
                        except asyncio.TimeoutError:
                            print(f"[{path}] ⚠️ 接收影像幀超時。")
                        except Exception as e:
                            print(f"[{path}] ⚠️ Track 錯誤: {e}")
                        finally:
                            if first_frame_received.empty(): await first_frame_received.put(False)

                url = f"{mediamtx_base_url}/{path}/whep"
                print(f"[{path}] 正在嘗試連接到 WebRTC: {url} ...")
                pc.addTransceiver("video", direction="recvonly")
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)

                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=pc.localDescription.sdp,
                                            headers={"Content-Type": "application/sdp"}) as resp:
                        if resp.status != 201:
                            print(f"[{path}] ❌ 連線失敗 {resp.status}，5秒後重試。")
                            await asyncio.sleep(5);
                            continue
                        answer_sdp = await resp.text()
                        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
                        print(f"[{path}] ✅ WebRTC SDP 交換成功！")

                try:
                    success = await asyncio.wait_for(first_frame_received.get(), timeout=15.0)
                    if not success:
                        print(f"[{path}] ⚠️ on_track 錯誤，重連。");
                        continue
                except asyncio.TimeoutError:
                    print(f"[{path}] ❌ 等待首幀逾時，重連。");
                    continue

                print(f"[{path}] 連線穩定，進入監控狀態...")
                while not shutdown_event.is_set() and pc.connectionState in ["connected", "connecting"]:
                    await asyncio.sleep(1)
                print(f"[{path}] 連線狀態變為 {pc.connectionState}，準備重連。")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{path}] ❌ 發生錯誤: {e}")
                traceback.print_exc()
            finally:
                if pc: await pc.close()
                if not shutdown_event.is_set():
                    print(f"[{path}] 連線已關閉，5秒後重連...")
                    await asyncio.sleep(5)

    loop.run_until_complete(receiver_task())


def webcam_receiver_thread(camera_index, frame_queue, shutdown_event):
    print(f"🎥 正在啟動本地攝像頭 #{camera_index} ...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"錯誤：無法開啟攝像頭 #{camera_index}")
        return
    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        if not frame_queue.full(): frame_queue.put_nowait(frame)
        time.sleep(0.01)
    cap.release()
    print(f"📷 攝像頭執行緒 #{camera_index} 已結束。")


# --- 3. 主程式 ---
def main():
    trip_id = f"trip_{uuid.uuid4().hex[:8]}"
    print(f"====== 本次行程 ID: {trip_id} ======")

    # 【新增功能 1】啟動時通知後端 (讓網頁儀表板亮紅燈)
    notify_backend(RECORDING_STATUS_ENDPOINT, {"session_id": trip_id, "status": "start"})

    shutdown_event = threading.Event()
    frame_queues, receiver_threads = {}, []

    if INPUT_SOURCE_MODE == "WEBRTC":
        active_paths = stream_paths
        print(f"--- 啟動 WebRTC 模式，處理串流: {active_paths} ---")
        for path in active_paths:
            frame_queues[path] = Queue(maxsize=5)
            thread = threading.Thread(target=webrtc_receiver_thread, args=(path, frame_queues[path], shutdown_event),
                                      daemon=True)
            receiver_threads.append(thread);
            thread.start()
    elif INPUT_SOURCE_MODE == "WEBCAM":
        active_paths = [f"webcam{i}" for i in local_camera_indices]
        print(f"--- 啟動 Webcam 模式: {local_camera_indices} ---")
        for i, path in zip(local_camera_indices, active_paths):
            frame_queues[path] = Queue(maxsize=5)
            thread = threading.Thread(target=webcam_receiver_thread, args=(i, frame_queues[path], shutdown_event),
                                      daemon=True)
            receiver_threads.append(thread);
            thread.start()
    else:
        print(f"錯誤：未知的 INPUT_SOURCE_MODE: {INPUT_SOURCE_MODE}");
        return

    print("--- 正在載入所有 AI 模型... ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = YOLO(YOLO_MODEL_PATH)
    with open(YNET_CONFIG_PATH, 'r', encoding='utf-8') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    obs_len = params['obs_len']
    ynet_model = YNet(obs_len=obs_len, pred_len=params['pred_len'], params=params)
    ynet_model.load(YNET_MODEL_PATH)
    ynet_model.model.to(device).eval()
    checkpoint = torch.load(SEG_MODEL_PATH, map_location=device, weights_only=False)
    seg_model = modeling.deeplabv3plus_mobilenet(num_classes=19, output_stride=16)
    seg_model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state'].items()})
    seg_model.to(device).eval()
    tracker = ByteTrack()
    input_template = torch.Tensor(create_dist_mat(size=2000)).to(device)
    print("--- 所有模型載入完成！ ---")

    all_track_histories = {path: defaultdict(lambda: deque(maxlen=obs_len)) for path in active_paths}
    all_track_predictions = {path: {} for path in active_paths}
    all_frame_idx = {path: 0 for path in active_paths}
    all_cached_seg_maps = {path: None for path in active_paths}
    last_danger_notify_time = {path: 0 for path in active_paths}
    video_writers = {}

    print("\n--- 等待所有影像來源的第一幀，最多等待 30 秒... ---")
    initial_frames = {}
    time.sleep(2)
    for path in active_paths:
        try:
            print(f"正在等待 [{path}] 的第一幀...")
            frame = frame_queues[path].get(timeout=30)
            initial_frames[path] = frame
            print(f"✅ 成功接收到 [{path}] 的第一幀！")
        except Empty:
            print(f"❌ 警告：等待 [{path}] 的第一幀超時，將忽略此串流。")

    active_paths = list(initial_frames.keys())
    if not active_paths:
        print("錯誤：沒有任何影像來源成功連接。程式即將退出。")
        shutdown_event.set()

    try:
        while not shutdown_event.is_set():
            for path in active_paths:
                if path in initial_frames:
                    frame_orig = initial_frames.pop(path)
                else:
                    try:
                        frame_orig = frame_queues[path].get_nowait()
                    except Empty:
                        continue

                window_name = f"Intelligent Recorder - {path}"
                track_histories = all_track_histories[path]
                track_predictions = all_track_predictions[path]
                frame_idx = all_frame_idx[path]
                cached_seg_map = all_cached_seg_maps[path]

                if path not in video_writers and SAVE_VIDEO:
                    height, width, _ = frame_orig.shape
                    save_path = os.path.join(RECORD_OUTPUT_DIR,
                                             f"{path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                    video_writers[path] = {
                        'writer': VideoWriterThread(save_path, (width, height)),
                        'start_time': time.time(),
                        'path': save_path,
                        'danger_zone': np.array(
                            [[int(width * 0.25), int(height * 0.6)], [int(width * 0.75), int(height * 0.6)],
                             [int(width * 0.95), height - 1], [int(width * 0.05), height - 1]], np.int32)
                    }
                    video_writers[path]['writer'].start()

                canvas = frame_orig.copy()
                all_frame_idx[path] += 1
                frame_model_size = cv2.resize(frame_orig, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))

                results = yolo_model(frame_model_size, verbose=False, classes=CLASSES_TO_TRACK)
                detections = results[0].boxes.data.cpu().numpy()
                tracked_objects = tracker.update(detections, frame_model_size)
                active_track_ids = {int(obj[4]) for obj in tracked_objects}
                track_predictions = {tid: pred for tid, pred in track_predictions.items() if tid in active_track_ids}

                if frame_idx % SEGMENTATION_INTERVAL == 0 or cached_seg_map is None:
                    img_rgb = cv2.cvtColor(frame_model_size, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
                    mean, std = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1), torch.tensor(
                        [0.229, 0.224, 0.225], device=device).view(3, 1, 1)
                    seg_input_tensor = ((img_tensor.to(device) - mean) / std).unsqueeze(0)
                    with torch.no_grad():
                        seg_logits = seg_model(seg_input_tensor)
                        if isinstance(seg_logits, dict): seg_logits = seg_logits['out']
                        cached_seg_map = torch.argmax(seg_logits.squeeze(), dim=0).cpu().numpy()
                    all_cached_seg_maps[path] = cached_seg_map

                tracks_to_predict = []
                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = obj[:5]
                    track_histories[int(track_id)].append([(x1 + x2) / 2, y2])
                    if len(track_histories[int(track_id)]) == obs_len:
                        tracks_to_predict.append(int(track_id))

                if tracks_to_predict:
                    with torch.no_grad():
                        num_to_predict = len(tracks_to_predict)
                        batch_hist = torch.from_numpy(
                            np.array([list(track_histories[tid]) for tid in tracks_to_predict])).float().to(device)
                        vel = batch_hist[:, 1:] - batch_hist[:, :-1]
                        obs_vel = torch.cat([torch.zeros((num_to_predict, 1, 2), device=device), vel], dim=1)
                        acc = obs_vel[:, 1:] - obs_vel[:, :-1]
                        obs_acc = torch.cat([torch.zeros((num_to_predict, 1, 2), device=device), acc], dim=1)
                        h_ynet, w_ynet = MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH
                        seg_map_onehot = F.one_hot(torch.from_numpy(cached_seg_map).long().to(device), 19)
                        seg_map_batch = seg_map_onehot.permute(2, 0, 1).float().unsqueeze(0).repeat(num_to_predict, 1,
                                                                                                    1, 1)
                        vel_map = torch.stack([torch.stack(
                            [create_arrow_heatmap(h_ynet, w_ynet, c[0], c[1], v[0], v[1], device=device) for c, v in
                             zip(batch_hist[:, i, :], obs_vel[:, i, :])]) for i in range(obs_len)], dim=1)
                        acc_map = torch.stack([torch.stack(
                            [create_arrow_heatmap(h_ynet, w_ynet, c[0], c[1], a[0], a[1], device=device) for c, a in
                             zip(batch_hist[:, i, :], obs_acc[:, i, :])]) for i in range(obs_len)], dim=1)
                        features = ynet_model.model.pred_features(torch.cat([seg_map_batch, vel_map, acc_map], dim=1))
                        pred_waypoint = ynet_model.model.pred_goal(features)[:, params['waypoints']]
                        pred_waypoint_sm = ynet_model.model.sigmoid(pred_waypoint / params['temperature'])
                        goal_samples = sampling(pred_waypoint_sm[:, -1:],
                                                num_samples=params.get('num_goals', 20)).permute(2, 0, 1, 3)
                        goal_scores = torch.stack([torch.stack([pred_waypoint_sm[
                                                                    i, -1, torch.clamp(g[i, 0, 1].long(), 0,
                                                                                       h_ynet - 1), torch.clamp(
                                                                        g[i, 0, 0].long(), 0, w_ynet - 1)] for i in
                                                                range(num_to_predict)]) for g in goal_samples])
                        future_samples = []
                        for waypoint in goal_samples:
                            waypoint_map = get_patch(input_template, waypoint.reshape(-1, 2).cpu().numpy(), h_ynet,
                                                     w_ynet).reshape([-1, 1, h_ynet, w_ynet])
                            traj_input = [torch.cat([feat,
                                                     F.interpolate(waypoint_map, size=feat.shape[2:], mode='bilinear',
                                                                   align_corners=False)], dim=1) for feat in features]
                            future_samples.append(ynet_model.model.softargmax(ynet_model.model.pred_traj(traj_input)))
                        future_samples = torch.stack(future_samples)
                        best_indices = torch.argmax(goal_scores, dim=0)
                        best_future = future_samples.permute(1, 0, 2, 3)[torch.arange(num_to_predict), best_indices]
                        for i, track_id in enumerate(tracks_to_predict):
                            track_predictions[track_id] = best_future[i].cpu().numpy()

                orig_h, orig_w = frame_orig.shape[:2]
                w_scale, h_scale = orig_w / MODEL_INPUT_WIDTH, orig_h / MODEL_INPUT_HEIGHT

                danger_zone_poly = video_writers.get(path, {}).get('danger_zone')
                is_danger = False
                if danger_zone_poly is not None:
                    is_danger = any(
                        cv2.pointPolygonTest(danger_zone_poly, (int(p[0]), int(p[1])), False) >= 0 for tid in
                        track_predictions for p in (track_predictions[tid] * [w_scale, h_scale]))

                    if is_danger and (time.time() - last_danger_notify_time[path] > 15):
                        last_danger_notify_time[path] = time.time()
                        print(f"[{path}] 偵測到危險！發送通知並冷卻 15 秒。")

                        # 【新增功能 2】 發送真實危險警報給後端 (儀表板會跳出)
                        event_timestamp_str = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                        danger_data = {
                            "trip_id": trip_id,
                            "event_type": "軌跡預測警告",
                            "description": f"鏡頭 [{path}] 偵測到有物體軌跡進入危險區域！",
                            "timestamp": event_timestamp_str
                        }
                        threading.Thread(target=notify_backend, args=(NOTIFY_DANGER_ENDPOINT, danger_data)).start()

                    zone_color = (0, 0, 255) if is_danger else (0, 255, 0)
                    cv2.polylines(canvas, [danger_zone_poly], True, zone_color, 3)

                for obj in tracked_objects:
                    x1, y1, x2, y2, tid = [int(p) for p in obj[:5]]
                    x1_o, y1_o, x2_o, y2_o = int(x1 * w_scale), int(y1 * h_scale), int(x2 * w_scale), int(y2 * h_scale)
                    cv2.rectangle(canvas, (x1_o, y1_o), (x2_o, y2_o), (0, 255, 0), 2)
                    cv2.putText(canvas, f"ID:{tid}", (x1_o, y1_o - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    obs_orig_defined = False
                    if tid in track_histories and len(track_histories[tid]) > 1:
                        obs_orig = (np.array(track_histories[tid]) * [w_scale, h_scale]).astype(np.int32)
                        obs_orig_defined = True
                        for k in range(len(obs_orig) - 1):
                            cv2.line(canvas, tuple(obs_orig[k]), tuple(obs_orig[k + 1]), (255, 100, 0), 2)
                    if tid in track_predictions and obs_orig_defined:
                        pred_orig = (track_predictions[tid] * [w_scale, h_scale]).astype(int)
                        full_pred = np.vstack([obs_orig[-1], pred_orig])
                        for k in range(len(full_pred) - 1):
                            cv2.line(canvas, tuple(full_pred[k]), tuple(full_pred[k + 1]), (0, 0, 255), 2)

                cv2.imshow(window_name, canvas)
                if path in video_writers:
                    video_writers[path]['writer'].add_frame_to_queue(canvas)

            # --- 【新增功能 3】按鍵控制 ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                shutdown_event.set()
                break

            # 按 'd' 鍵手動測試危險警報
            elif key == ord('d'):
                print("🔥 手動觸發測試警報！")
                test_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                test_data = {
                    "trip_id": trip_id,
                    "event_type": "手動測試",
                    "description": "使用者手動觸發測試警報",
                    "timestamp": test_time
                }
                threading.Thread(target=notify_backend, args=(NOTIFY_DANGER_ENDPOINT, test_data)).start()

    finally:
        print(f"====== 行程 {trip_id} 已結束 ======")
        shutdown_event.set()
        for thread in receiver_threads:
            thread.join(timeout=2)
        for path in video_writers:
            vw_data = video_writers[path]
            vw_data['writer'].stop()
            vw_data['writer'].join()
            video_filename = os.path.basename(vw_data['path'])
            relative_path = os.path.join(VIDEO_FILES_DIR, video_filename).replace("\\", "/")
            upload_data = {"trip_id": trip_id, "path": path, "relative_path": relative_path,
                           "title": f"PiCam錄影 - {path}", "date": datetime.now().strftime('%m/%d'),
                           "description": f"時長約 {round(time.time() - vw_data['start_time'])} 秒。", }
            notify_backend(UPLOAD_VIDEO_ENDPOINT, upload_data)

        # 通知後端結束錄影
        notify_backend(RECORDING_STATUS_ENDPOINT, {"session_id": trip_id, "status": "end"})

        cv2.destroyAllWindows()
        print("程式已結束。")


if __name__ == "__main__":
    main()