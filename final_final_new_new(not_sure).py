# final_stable_recorder.py (基于 weqeqwe.py 成功架构，包含启动同步的最终完整版)

import cv2
import os
import sys
import numpy as np
import torch
import yaml  # 讀取 YAML 配置檔
from aiortc import RTCSessionDescription, RTCPeerConnection # aiortc是 WebRTC 的 Python 實作
#RTCPeerConnection 用於建立 WebRTC 連線

from ultralytics import YOLO # YOLOv8 物件偵測模型
from boxmot.trackers.bytetrack.bytetrack import ByteTrack # ByteTrack 追蹤器
from collections import defaultdict, deque # 用於儲存追蹤歷史

import time
import uuid# 用於產生唯一的行程 ID
import requests # 用於與 Flask 後端通訊

#asyncio 與 aiohttp 用於非同步 WebRTC 影像接收
#因為 WebRTC 需要非同步處理來有效率地接收影像串流


#webrtc 他的處理步驟包含 1.建立 RTCPeerConnection 2.交換 SDP 3.接收影像 Track
#asynicio 主要處理的部分是等待影像幀的接收
#http   則是用來與 mediamtx 伺服器交換 SDP
import asyncio # 非同步處理
import aiohttp # 非同步 HTTP 請求
from aiortc import RTCConfiguration, RTCIceServer  #這邊則是用來設定 ICE 伺服器
# 甚麼是ICE 伺服器呢? ICE 伺服器用於協助 WebRTC 連線的建立
#相當於中介伺服器 幫助雙方找到彼此 這裡的雙方 是指兩個 WebRTC 客戶端 一個pc 一個meadiamtx
#configuraion 是用來設定 RTCPeerConnection 的參數
#我們在這裡設定 ICE 伺服器 讓 WebRTC 能夠順利穿越 NAT 與防火牆
#這樣才能成功建立 P2P 連線
#rtcsessiondescription 則是用來封裝 SDP 資訊的物件
#SDP 是 WebRTC 用來描述多媒體連線參數的格式
#它包含了編解碼器 網路位址等資訊




import traceback # 用於錯誤追蹤
import threading # 多執行緒處理

from queue import Queue, Empty # 線程安全的佇列
#這個線程就是threading.Thread 他能夠讓我們在背景執行任務
#Queue 是線程安全的佇列 用於在不同線程間傳遞資料

import torch.nn.functional as F #pytorch 的函式庫 用於張量操作 張量是多維陣列 類似 numpy 的陣列
from datetime import datetime # 用於取得目前時間


import time
# --- 1. 路径与配置 ---


try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) #當前檔案路徑
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..')) #專案根目錄 #就是上一層目錄stream_yolo
    YNET_PROJECT_PATH = os.path.join(PROJECT_ROOT, 'Ynet_kitti_tracking-master') #Y-Net 專案路徑 #stream_yolo/Ynet_kitti_tracking-master
    if YNET_PROJECT_PATH not in sys.path: #如果 YNET_PROJECT_PATH 不在系統路徑中
        sys.path.append(YNET_PROJECT_PATH) #加入系統路徑 系統路徑
    from model import YNet #Y-Net 模型
    from network import modeling #
    from utils.image_utils import create_arrow_heatmap, get_patch, create_dist_mat, sampling #Y-Net 工具函式
except ImportError as e:
    print(f"错误：无法导入 Y-Net 相关模组: {e}")
    print(f"请确保 'Ynet_kitti_tracking-master' 文件夹与您的专案文件夹位于同一层目录下。")
    sys.exit(1)

# --- 配置選項 --- 可選擇 WebRTC 或 本地 Webcam 作為輸入來源
INPUT_SOURCE_MODE = "WEBRTC"  # 可选项: "WEBRTC" 或 "WEBCAM"

# --- 通用配置 ---
VIDEO_FILES_DIR = "videos"  #影片儲存目錄
RECORD_OUTPUT_DIR = os.path.join(CURRENT_DIR, "web_server", VIDEO_FILES_DIR) #當前位置的 web_server/videos
os.makedirs(RECORD_OUTPUT_DIR, exist_ok=True) #確保目錄存在
print(f"影片将储存至: {RECORD_OUTPUT_DIR}") #列印影片儲存目錄

FLASK_BACKEND_URL = "http://192.168.196.73:5000"  #Flask 後端伺服器 URL 這zerotier給的 所以只要在同一個網路就可以連到 android只要zerotier就可以連到這個ip 把gps傳回去
#這個 ip 是後端伺服器的位址

RECORDING_STATUS_ENDPOINT = f"{FLASK_BACKEND_URL}/recording_status"  #POST /recording_status
UPLOAD_VIDEO_ENDPOINT = f"{FLASK_BACKEND_URL}/upload_recorded_video" #POST /upload_recorded_video
NOTIFY_DANGER_ENDPOINT = f"{FLASK_BACKEND_URL}/notify_danger" #POST /notify_danger

YOLO_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'yolov8m.pt') #YOLOv8 模型路徑
YNET_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'pretrained_models/kitti_ynet_baseline_s13_best.pt') #Y-Net 模型路徑
SEG_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'segmentation_models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')#語義分割模型路徑
YNET_CONFIG_PATH = os.path.join(YNET_PROJECT_PATH, r'kitti_train_data/config/kitti.yaml') #Y-Net 配置檔路徑

SEGMENTATION_INTERVAL = 1 #每 N 幀執行一次語義分割
MODEL_INPUT_WIDTH = 640 #YOLO 與 Y-Net 模型輸入尺寸
MODEL_INPUT_HEIGHT = 192 #YOLO 與 Y-Net 模型輸入尺寸
CLASSES_TO_TRACK = [0, 1, 2, 3, 5, 7] # 增加了 0: person
# SAVE_VIDEO = True #是否儲存錄影影片  <-- 這個變數在新邏輯中不再需要

# --- WebRTC 配置 ---
mediamtx_base_url = "http://192.168.196.73:8889" # 远端树莓派的# 192.168.196.73  ZeroTier IP   #WIFI 10.21.78.41:8889
stream_paths = ["cam0", "cam1"]

# --- 本地 Webcam 配置 ---
local_camera_indices = [0] #本地攝像頭

# ---寫入影片的執行緒--- (保持您原有的版本，完全不變)
class VideoWriterThread(threading.Thread): #影片寫入執行緒
    def __init__(self, output_path, frame_size, fps=10.0):
        super().__init__() #呼叫父類別threading.Thread的初始化方法
        self.daemon = True #設置為守護線程 作用是當主線程結束時 自動結束這個線程 這樣就不用寫 .join() 來等待線程結束
        self.output_path, self.frame_size, self.fps = output_path, frame_size, fps #影片輸出路徑 幀尺寸 幀率
        self.write_queue = Queue(maxsize=120) #寫入佇列 最多120幀
        self.running = True #執行狀態
        self.writer = None #影片寫入器
    def run(self): #為什麼一定要覆寫 run 方法? 因為 threading.Thread 的 run 方法是空的
        #threading.Thread 在新執行緒內執行的「入口函式」。覆寫 run() 的目的就是把你要在那個新執行緒裡做的工作寫進去
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1') #使用 AVC1 編碼器
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)#建立影片寫入器
            if not self.writer.isOpened(): raise IOError("AVC1 failed") #檢查是否成功開啟
            print(f"影片寫入執行緒(avc1)已啟動: {os.path.basename(self.output_path)}")
        except Exception:
            print(f"警告: AVC1 編碼器不可用, 降級至 MP4V for {os.path.basename(self.output_path)}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)#建立影片寫入器
        while self.running or not self.write_queue.empty(): #持續寫入直到停止且佇列清空 佇列是
            try:
                frame = self.write_queue.get(timeout=1) #等待1秒取出一幀
                if self.writer: self.writer.write(frame) #寫入影片
            except Empty: continue
        if self.writer: self.writer.release() #釋放影片寫入器
        print(f"影片 {os.path.basename(self.output_path)} 寫入完成。")
    def add_frame_to_queue(self, frame): #加入幀到佇列
        if not self.write_queue.full(): self.write_queue.put_nowait(frame) #非阻塞加入
    def stop(self): self.running = False#停止寫入

# --- notify_backend 函式 (保持您原有的版本，完全不變) ---
def notify_backend(endpoint, data): #通知 Flask 後端
    try:
        response = requests.post(endpoint, json=data, timeout=5) #POST 請求
        response.raise_for_status() #檢查是否成功
        print(f"成功通知後端 {os.path.basename(endpoint)}，狀態碼: {response.status_code}") #列印成功訊息
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"錯誤：通知後端 {os.path.basename(endpoint)} 失敗: {e}")
        return None

# --- webrtc_receiver_task 函式 (保持您原有的版本，完全不變) ---
async def webrtc_receiver_task(path, frame_queue, shutdown_event):
    # ... (您所有的 webrtc_receiver_task 程式碼都保持不變) ...
    pc = RTCPeerConnection()
    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            while not shutdown_event.is_set():
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=10)
                    if not frame_queue.full():
                        frame_queue.put_nowait(frame.to_ndarray(format="bgr24"))
                except asyncio.TimeoutError:
                    print(f"[{path}] 接收影像幀超時。")
                    break
                except Exception: break
    try:
        url = f"{mediamtx_base_url}/{path}/whep"
        print(f"[{path}] 正在連接到 WebRTC: {url} ...")
        pc.addTransceiver("video", direction="recvonly")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=pc.localDescription.sdp, headers={"Content-Type": "application/sdp"}) as resp:
                if resp.status != 201:
                    print(f"[{path}] 連線失敗，狀態碼: {resp.status}")
                    return
                answer_sdp = await resp.text()
                await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
                print(f"[{path}] WebRTC 連線成功！")
        while not shutdown_event.is_set():
            await asyncio.sleep(0.5)
    except asyncio.CancelledError: pass
    finally:
        await pc.close()
        print(f"[{path}] WebRTC 連線已關閉。")

# --- webrtc_receiver_thread & webcam_receiver_thread 函式 (保持您原有的版本，完全不變) ---
def webrtc_receiver_thread(path, frame_queue, shutdown_event):
    # ... (您所有的 webrtc_receiver_thread 程式碼都保持不變) ...
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # ... (其餘 receiver_task 的內容也完全不變) ...
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
                    if track.kind == "video":
                        try:
                            first_frame = await asyncio.wait_for(track.recv(), timeout=15.0)
                            if not frame_queue.full(): frame_queue.put_nowait(first_frame.to_ndarray(format="bgr24"))
                            await first_frame_received.put(True)
                            while not shutdown_event.is_set():
                                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                                if not frame_queue.full(): frame_queue.put_nowait(frame.to_ndarray(format="bgr24"))
                        except (asyncio.TimeoutError, Exception):
                            if first_frame_received.empty(): await first_frame_received.put(False)
                url = f"{mediamtx_base_url}/{path}/whep"
                pc.addTransceiver("video", direction="recvonly")
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=pc.localDescription.sdp, headers={"Content-Type": "application/sdp"}) as resp:
                        if resp.status != 201:
                            await asyncio.sleep(5); continue
                        answer_sdp = await resp.text()
                        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
                success = await asyncio.wait_for(first_frame_received.get(), timeout=15.0)
                if not success: continue
                while not shutdown_event.is_set() and pc.connectionState in ["connected", "connecting"]:
                    await asyncio.sleep(1)
            except (asyncio.CancelledError, Exception): pass
            finally:
                if pc: await pc.close()
                if not shutdown_event.is_set(): await asyncio.sleep(5)
    loop.run_until_complete(receiver_task())


def webcam_receiver_thread(camera_index, frame_queue, shutdown_event):
    # ... (您所有的 webcam_receiver_thread 程式碼都保持不變) ...
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的計算裝置: {device}")
    trip_id = f"trip_{uuid.uuid4().hex[:8]}"
    print(f"====== 本次行程 ID: {trip_id} ======")
    notify_backend(RECORDING_STATUS_ENDPOINT, {"session_id": trip_id, "status": "start"})

    shutdown_event = threading.Event()
    frame_queues, receiver_threads = {}, []

    active_paths = [] # 為了讓 IDE 不報錯，先初始化
    if INPUT_SOURCE_MODE == "WEBRTC":
        active_paths = stream_paths
        print(f"--- 啟動 WebRTC 模式，處理串流: {active_paths} ---")
        for path in active_paths:
            frame_queues[path] = Queue(maxsize=30)
            thread = threading.Thread(target=webrtc_receiver_thread, args=(path, frame_queues[path], shutdown_event), daemon=True)
            receiver_threads.append(thread); thread.start()

    elif INPUT_SOURCE_MODE == "WEBCAM":
        active_paths = [f"webcam{i}" for i in local_camera_indices]
        print(f"--- 啟動本地 Webcam 模式，處理鏡頭: {local_camera_indices} ---")
        for i, path in zip(local_camera_indices, active_paths):
            frame_queues[path] = Queue(maxsize=30)  # 保持與 WebRTC 模式一致的佇列大小
            thread = threading.Thread(target=webcam_receiver_thread, args=(i, frame_queues[path], shutdown_event),
                                      daemon=True)
            receiver_threads.append(thread)
            thread.start()
    else:
        print(f"錯誤：未知的 INPUT_SOURCE_MODE: {INPUT_SOURCE_MODE}"); return

    # --- 模型載入部分 (保持您原有的版本，完全不變) ---
    print("--- 正在載入所有 AI 模型... ---")
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
    tracker = ByteTrack(track_thresh=0.1,match_thresh=0.4) # 初始化 ByteTrack 追蹤器 #0.1是偵測門檻值 越低越靈敏 但也可能誤追蹤 0.6是匹配門檻值 越高越嚴格
    input_template = torch.Tensor(create_dist_mat(size=2000)).to(device)
    print("--- 所有模型載入完成！ ---")

    # --- 初始化 AI 分析與新版錄影相關的變數 ---
    all_track_histories = {path: defaultdict(lambda: deque(maxlen=obs_len)) for path in active_paths}
    all_track_predictions = {path: {} for path in active_paths}
    all_frame_idx = {path: 0 for path in active_paths}
    all_cached_seg_maps = {path: None for path in active_paths}
    last_danger_notify_time = {path: 0 for path in active_paths}

    # 【【【 核心修改 1：初始化事件錄影變數 】】】
    PRE_EVENT_SECONDS = 5
    POST_EVENT_SECONDS = 60
    EVENT_COOLDOWN_SECONDS = 30  # <--- 【【【 新增 】】
    frame_buffers = {path: deque(maxlen=int(PRE_EVENT_SECONDS * 15)) for path in active_paths} # 假設  FPS
    event_recording_status = {
        path: {"writer": None, "stop_time": 0} for path in active_paths
    }

    # --- 等待第一幀 (保持您原有的版本，完全不變) ---
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
        return

    # --- 主迴圈 ---
    try:
        while not shutdown_event.is_set():
            for path in active_paths:
                if path in initial_frames:
                    frame_orig = initial_frames.pop(path)
                else:
                    try:
                        frame_orig = None
                        # 如果佇列中的幀超過一定數量 (例如 5)，就清空舊的，只拿最新的
                        if frame_queues[path].qsize() > 5:
                            # print(f"[{path}] 處理延遲，正在丟棄舊幀...")
                            while not frame_queues[path].empty():
                                try:
                                    frame_orig = frame_queues[path].get_nowait()
                                except Empty:
                                    break
                        else:
                            frame_orig = frame_queues[path].get_nowait()

                        if frame_orig is None:
                            continue
                    except Empty:
                        continue

                # --- 從這裡開始，AI 分析和繪圖的程式碼都保持不變 ---
                window_name = f"Intelligent Recorder - {path}"
                track_histories = all_track_histories[path]
                track_predictions = all_track_predictions[path]
                frame_idx = all_frame_idx[path]
                cached_seg_map = all_cached_seg_maps[path]

                canvas = frame_orig.copy()
                all_frame_idx[path] += 1
                frame_model_size = cv2.resize(frame_orig, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))

                # ... (所有 YOLO, ByteTrack, 分割, 軌跡預測的程式碼都保持不變) ...
                results = yolo_model(frame_model_size, conf=0.1,verbose=False, classes=CLASSES_TO_TRACK)
                detections = results[0].boxes.data.cpu().numpy()
                tracked_objects = tracker.update(detections, frame_model_size)
                active_track_ids = {int(obj[4]) for obj in tracked_objects}
                track_predictions = {tid: pred for tid, pred in track_predictions.items() if tid in active_track_ids}
                if frame_idx % SEGMENTATION_INTERVAL == 0 or cached_seg_map is None:
                    img_rgb = cv2.cvtColor(frame_model_size, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb.astype(np.float32)/255.0).permute(2,0,1)
                    mean, std = torch.tensor([0.485,0.456,0.406],device=device).view(3,1,1), torch.tensor([0.229,0.224,0.225],device=device).view(3,1,1)
                    seg_input_tensor = ((img_tensor.to(device) - mean)/std).unsqueeze(0)
                    with torch.no_grad():
                        seg_logits = seg_model(seg_input_tensor)
                        if isinstance(seg_logits, dict): seg_logits = seg_logits['out']
                        cached_seg_map = torch.argmax(seg_logits.squeeze(), dim=0).cpu().numpy()
                    all_cached_seg_maps[path] = cached_seg_map
                tracks_to_predict = []
                for obj in tracked_objects: ###
                    x1, y1, x2, y2, track_id = obj[:5]
                    track_histories[int(track_id)].append([(x1+x2)/2, y2])
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

                        # --- 這裡就是被刪掉的關鍵定義 ---
                        best_future = future_samples.permute(1, 0, 2, 3)[torch.arange(num_to_predict), best_indices]

                        for i, track_id in enumerate(tracks_to_predict):
                            track_predictions[track_id] = best_future[i].cpu().numpy()

                orig_h, orig_w = frame_orig.shape[:2]
                w_scale, h_scale = orig_w/MODEL_INPUT_WIDTH, orig_h/MODEL_INPUT_HEIGHT

                # 【【【 核心修改 2：用新的事件錄影邏輯替換舊的錄影和通知邏輯 】】】
                # 1. 將每一幀都先存入缓冲区
                frame_buffers[path].append(canvas)

                # 2. 動態定義危險區域並判斷
                height, width, _ = canvas.shape
                danger_zone_poly = np.array([
                    [int(width * 0.25), int(height * 0.6)], [int(width * 0.75), int(height * 0.6)],
                    [int(width * 0.95), height - 1], [int(width * 0.05), height - 1]
                ], np.int32)
                is_danger = any(cv2.pointPolygonTest(danger_zone_poly, (int(p[0]), int(p[1])), False) >= 0 for tid in
                                track_predictions for p in (track_predictions[tid] * [w_scale, h_scale]))

                # 3. 核心錄影與通知邏輯
                current_time = time.time()
                status = event_recording_status[path]

                if is_danger:
                    if status["writer"] is None:
                        print(f"[{path}] 🚨 危險事件觸發！開始錄製...")
                        save_path = os.path.join(RECORD_OUTPUT_DIR, f"EVENT_{path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                        status["writer"] = VideoWriterThread(save_path, (width, height))
                        status["writer"].start()
                        for frame_in_buffer in list(frame_buffers[path]):
                            status["writer"].add_frame_to_queue(frame_in_buffer)

                        if current_time - last_danger_notify_time.get(path, 0) > 10:
                            last_danger_notify_time[path] = current_time
                            print(f"[{path}] 發送危險通知並冷卻 10 秒。")
                            event_timestamp_str = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                            danger_data = {"trip_id": trip_id, "event_type": "軌跡預測警告", "description": f"鏡頭 [{path}] 偵測到有物體軌跡進入危險區域！","timestamp": event_timestamp_str}
                            threading.Thread(target=notify_backend, args=(NOTIFY_DANGER_ENDPOINT, danger_data)).start()

                    status["stop_time"] = current_time + POST_EVENT_SECONDS


                elif status["writer"] is not None and current_time > status["stop_time"]:
                    print(f"[{path}] ✅ 事件結束，停止錄影。進入 {EVENT_COOLDOWN_SECONDS} 秒冷卻期。")
                    status["last_event_end_time"] = current_time  # <--- 【【【 新增 】】】
                    writer_to_stop = status["writer"]
                    writer_to_stop.stop()


                    # 在背景執行緒中處理影片上傳
                    def final_upload(writer, trip_id, path):
                        writer.join() # 等待影片寫入完成
                        video_filename = os.path.basename(writer.output_path)
                        relative_path = os.path.join(VIDEO_FILES_DIR, video_filename).replace("\\","/")
                        upload_data = { "trip_id": trip_id, "path": path, "relative_path": relative_path, "title": f"危險事件 - {path}", "date": datetime.now().strftime('%m/%d'), "description": f"偵測到危險事件的片段錄影。"}
                        notify_backend(UPLOAD_VIDEO_ENDPOINT, upload_data)

                    threading.Thread(target=final_upload, args=(writer_to_stop, trip_id, path)).start()
                    status["writer"] = None

                if status["writer"] is not None:
                    status["writer"].add_frame_to_queue(canvas)

                # 4. 繪製視覺效果 (這部分邏輯保持不變，只是現在它在新的錄影邏輯塊之後)
                zone_color = (0, 0, 255) if is_danger else (0, 255, 0)
                cv2.polylines(canvas, [danger_zone_poly], True, zone_color, 3)
                # if is_danger:
                    # text = "!!! WARNING !!!"
                    # text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 2, 3)
                    # cv2.putText(canvas, text, ((width - text_size[0]) // 2, int(height * 0.2)), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)

                # --- 繪製軌跡和框線的程式碼 (保持您原有的版本，完全不變) ---
                for obj in tracked_objects:
                    track_id = int(obj[4])
                    x1_m, y1_m, x2_m, y2_m = [int(p) for p in obj[:4]]

                    # 反向縮放追蹤框
                    x1_o, y1_o = int(x1_m * w_scale), int(y1_m * h_scale)
                    x2_o, y2_o = int(x2_m * w_scale), int(y2_m * h_scale)
                    cv2.rectangle(canvas, (x1_o, y1_o), (x2_o, y2_o), (0, 255, 0), 2)
                    cv2.putText(canvas, f"ID:{track_id}", (x1_o, y1_o - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                                2)

                    # b. 获取当前物件的所有相关资料
                    history_deq = track_histories.get(track_id)
                    prediction_arr = track_predictions.get(track_id)

                    # c. 先绘制历史轨迹 (蓝色)
                    obs_orig = None
                    if history_deq and len(history_deq) > 0:
                        history_np = np.array(history_deq)
                        # 【注意】history_deq 已经是模型坐标系，需要反向缩放
                        obs_orig = (history_np * [w_scale, h_scale]).astype(np.int32)
                        for k in range(len(obs_orig) - 1):
                            cv2.line(canvas, tuple(obs_orig[k]), tuple(obs_orig[k + 1]), (255, 100, 0), 2)

                    # d. 【双重验证与动态步长逻辑】
                    #    只有当【同时】有历史和预测时，才进行判断
                    if prediction_arr is not None and history_deq and len(history_deq) >= 2:

                        # --- I. 行为验证：检查是否“迴转” (在模型坐标系下进行) ---
                        current_pos_model = history_deq[-1]
                        prev_pos_model = history_deq[-2]
                        current_velocity_x = current_pos_model[0] - prev_pos_model[0]

                        predicted_end_point_model = prediction_arr[-1]
                        predicted_direction_x = predicted_end_point_model[0] - current_pos_model[0]

                        # is_reversing 的判断逻辑保持不变
                        is_reversing = current_velocity_x * predicted_direction_x < -1.0

                        # 如果是迴转，就直接跳过这个物体的预测绘制
                        if is_reversing:
                            continue

                        # --- II. 动态步长：如果不是迴转，再计算应该画多长 (在模型坐标系下进行) ---
                        current_x_model = current_pos_model[0]

                        distance_to_edge = min(current_x_model, MODEL_INPUT_WIDTH - current_x_model)

                        min_pred_steps = 3
                        # 信心比例：在中心为 1.0，在边缘为 0.0
                        confidence_ratio = np.clip(distance_to_edge / (MODEL_INPUT_WIDTH / 2.0), 0.0, 1.0)
                        dynamic_pred_len = int(min_pred_steps + (8 - min_pred_steps) * confidence_ratio)

                        # --- III. 截断并绘制 ---
                        pred_model_truncated = prediction_arr[:dynamic_pred_len]

                        # 将截断后的轨迹反向缩放回原始画布坐标
                        pred_orig = (pred_model_truncated * [w_scale, h_scale]).astype(int)

                        if obs_orig is not None and pred_orig.shape[0] > 0:
                            start_point = obs_orig[-1]
                            full_pred = np.vstack([start_point, pred_orig])
                            for k in range(len(full_pred) - 1):
                                cv2.line(canvas, tuple(full_pred[k]), tuple(full_pred[k + 1]), (0, 0, 255), 2)

                cv2.imshow(window_name, canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                shutdown_event.set()
                break
    finally:
        print(f"====== 行程 {trip_id} 已結束 ======")
        shutdown_event.set()
        for thread in receiver_threads:
            thread.join(timeout=2)

        # 【【【 核心修改 3：修改 finally 區塊 】】】
        # 移除舊的 video_writers 處理邏輯
        # 新增對 event_recording_status 的檢查
        for path in active_paths:
            status = event_recording_status.get(path)
            if status and status.get("writer") is not None:
                print(f"[{path}] 程式結束，強制停止並儲存正在進行的事件錄影...")
                writer_to_stop = status["writer"]
                writer_to_stop.stop()
                writer_to_stop.join() # 在主執行緒等待，確保影片寫完再結束
                # 這裡也可以觸發最後一次上傳
                video_filename = os.path.basename(writer_to_stop.output_path)
                relative_path = os.path.join(VIDEO_FILES_DIR, video_filename).replace("\\","/")
                upload_data = { "trip_id": trip_id, "path": path, "relative_path": relative_path, "title": f"危險事件 - {path}", "date": datetime.now().strftime('%m/%d'), "description": f"程式結束時儲存的片段錄影。"}
                notify_backend(UPLOAD_VIDEO_ENDPOINT, upload_data)

        notify_backend(RECORDING_STATUS_ENDPOINT, {"session_id": trip_id, "status": "end"})
        cv2.destroyAllWindows()
        print("程式已結束。")

if __name__ == "__main__":
    main()