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
INPUT_SOURCE_MODE = "WEBCAM"  # 可选项: "WEBRTC" 或 "WEBCAM"

# --- 通用配置 ---
VIDEO_FILES_DIR = "videos"  #影片儲存目錄
RECORD_OUTPUT_DIR = os.path.join(CURRENT_DIR, "web_server", VIDEO_FILES_DIR) #當前位置的 web_server/videos
os.makedirs(RECORD_OUTPUT_DIR, exist_ok=True) #確保目錄存在
print(f"影片将储存至: {RECORD_OUTPUT_DIR}") #列印影片儲存目錄

FLASK_BACKEND_URL = "http://192.168.196.207:5000"  #Flask 後端伺服器 URL 這zerotier給的 所以只要在同一個網路就可以連到 android只要zerotier就可以連到這個ip 把gps傳回去
RECORDING_STATUS_ENDPOINT = f"{FLASK_BACKEND_URL}/recording_status"  #POST /recording_status
UPLOAD_VIDEO_ENDPOINT = f"{FLASK_BACKEND_URL}/upload_recorded_video" #POST /upload_recorded_video
NOTIFY_DANGER_ENDPOINT = f"{FLASK_BACKEND_URL}/notify_danger" #POST /notify_danger

YOLO_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'yolov8m.pt') #YOLOv8 模型路徑
YNET_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'pretrained_models/kitti_ynet_baseline_s8_best.pt') #Y-Net 模型路徑
SEG_MODEL_PATH = os.path.join(YNET_PROJECT_PATH, 'segmentation_models/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')#語義分割模型路徑
YNET_CONFIG_PATH = os.path.join(YNET_PROJECT_PATH, r'kitti_train_data/config/kitti.yaml') #Y-Net 配置檔路徑

SEGMENTATION_INTERVAL = 2 #每 N 幀執行一次語義分割
MODEL_INPUT_WIDTH = 320 #YOLO 與 Y-Net 模型輸入尺寸
MODEL_INPUT_HEIGHT = 96 #YOLO 與 Y-Net 模型輸入尺寸
CLASSES_TO_TRACK = [0, 1, 2, 3, 5, 7] # 增加了 0: person
SAVE_VIDEO = True #是否儲存錄影影片

# --- WebRTC 配置 ---
mediamtx_base_url = "http://192.168.196.73:8889" # 远端树莓派的 ZeroTier IP   #WIFI  10.21.78.41:8889
stream_paths = ["cam0", "cam1"]

# --- 本地 Webcam 配置 ---
local_camera_indices = [0] #本地攝像頭

# ---寫入影片的執行緒---
class VideoWriterThread(threading.Thread): #影片寫入執行緒
    def __init__(self, output_path, frame_size, fps=20.0):
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






def notify_backend(endpoint, data): #通知 Flask 後端
    try:
        response = requests.post(endpoint, json=data, timeout=5) #POST 請求
        response.raise_for_status() #檢查是否成功
        print(f"成功通知後端 {os.path.basename(endpoint)}，狀態碼: {response.status_code}") #列印成功訊息
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"錯誤：通知後端 {os.path.basename(endpoint)} 失敗: {e}")
        return None






#建立 RTCPeerConnection 並交換 SDP 然後接收影像 Track
#詢問: 這邊到print(f"[{path}] WebRTC 連線已關閉。")  這樣還不夠嗎?
#答: 這段程式碼確實涵蓋了建立 WebRTC 連線並接收影像的基本流程
#不過在實際應用中 可能還需要考慮更多的錯誤處理 與重連機制
#例如如果連線中斷了 怎麼自動重連
#或者如果接收影像幀超時了 怎麼處理
#這些都可以根據實際需求來擴展這個基礎的接收任務

async def webrtc_receiver_task(path, frame_queue, shutdown_event): #非同步 WebRTC 接收任務
    pc = RTCPeerConnection() #建立 RTCPeerConnection


    @pc.on("track") #當接收到 Track 時觸發  這裡的 Track 就是影像串流 我這邊接收 mediamtx 伺服器傳來的影像串流
    async def on_track(track): #接收 Track 的回呼函式
        if track.kind == "video": #如果是影像 Track
            while not shutdown_event.is_set(): #持續接收直到關閉事件被設定 #shutdown_event 是 threading.Event 用於通知關閉
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=10) # 這裡的 track.recv() 是非同步方法 用於接收影像幀
                    if not frame_queue.full():
                        frame_queue.put_nowait(frame.to_ndarray(format="bgr24")) #將影像幀轉為 numpy 陣列 並放入佇列 為什麼要這麼做? 因為主線程會從這個佇列取出影像幀進行處理
                except asyncio.TimeoutError:
                    print(f"[{path}] 接收影像幀超時。")
                    break
                except Exception: break


    try:
        # === STEP 1 建立 WebRTC Offer ===
        url = f"{mediamtx_base_url}/{path}/whep" #WebRTC 連線 URL #192.168.196.73:8889/cam0/whep或cam1
        print(f"[{path}] 正在連接到 WebRTC: {url} ...")
        pc.addTransceiver("video", direction="recvonly") #添加影像接收器 #direction="recvonly" 表示只接收影像
        offer = await pc.createOffer() #建立 Offer SDP 這裡就是第二步 前面建立 RTCPeerConnection
        await pc.setLocalDescription(offer) #設定本地描述 為什麼要設定本地描述? 因為我們要把這個 Offer SDP 發送給 mediamtx 伺服器



        # === STEP 2. 傳送 SDP 給 MediaMTX（WHEP API） ===
        async with aiohttp.ClientSession() as session: #建立非同步 HTTP Session
            async with session.post(url, data=pc.localDescription.sdp, headers={"Content-Type": "application/sdp"}) as resp: #發送 POST 請求 交換 SDP
                if resp.status != 201: #檢查回應狀態碼 201 表示成功建立 WebRTC 連線
                    print(f"[{path}] 連線失敗，狀態碼: {resp.status}")
                    return
                answer_sdp = await resp.text() #取得 Answer SDP
                await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer")) #設定遠端描述 為什麼要設定遠端描述? 因為我們要告訴 RTCPeerConnection 對方的連線參數
                print(f"[{path}] WebRTC 連線成功！")


        # === STEP 3. 持續保持連線 ===
        while not shutdown_event.is_set():
            await asyncio.sleep(0.5)
    except asyncio.CancelledError: pass
    finally:
        await pc.close()
        print(f"[{path}] WebRTC 連線已關閉。")








#--- 2. WebRTC 接收執行緒 --- 這是背景執行緒 用於接收 WebRTC 影像串流 因為 WebRTC 需要非同步處理
#什麼叫非同步處理呢? 就是說我們不會阻塞主線程 等待影像幀的到來
#而是使用 asyncio 事件迴圈來處理影像幀的接收     asyncio是 Python 的非同步 I/O 框架
#IO 指的是輸入輸出操作 例如網路請求 檔案讀寫等
#這樣主線程就可以繼續執行其他任務 比如影像處理與顯示

#為什麼要有這個 上面不是就有連線了嗎?
#答: 這個 webrtc_receiver_thread 函式是對上面 webrtc_receiver_task 的一個封裝
#它在一個獨立的執行緒中運行非同步的 WebRTC 接收任務
#這樣可以讓主線程專注於影像處理與顯示 而不會被 WebRTC 的非同步操作阻塞


#在背景執行緒中運行非同步的 WebRTC 接收任務
def webrtc_receiver_thread(path, frame_queue, shutdown_event):  #frame_queqe是path兩個鏡頭

    #WebRTC 接收執行緒，包含自動重連機制
    #重作吧...應該算


    # 為每個執行緒建立並設定獨立的 asyncio 事件迴圈
    loop = asyncio.new_event_loop()  # 在背景執行非同步事件迴圈
    asyncio.set_event_loop(loop) #設定當前執行緒的事件迴圈


    #接單流程
    async def receiver_task(): #非同步接收任務
        while not shutdown_event.is_set():  # 這裡是自動重連的關鍵  當0的時候 持續工作直到關門
            pc = None
            try:
                # 建立新連線的程式碼
                ice_servers = [
                    RTCIceServer(urls=["stun:stun.l.google.com:19302"]) #使用 Google 的公共 STUN 伺服器 #協助 NAT 穿越
                ]
                config = RTCConfiguration(iceServers=ice_servers) #建立 RTC 配置

                # 啟用 RTCP 反饋機制，讓客戶端可以請求關鍵幀 (PLI/FIR)
                pc = RTCPeerConnection(configuration=config)
                pc.RTCP_REPORTS_DEFAULT = True  # 確保 RTCP 報告是啟用的

                # 使用一個非同步的 Queue 來標記 on_track 是否成功接收到第一幀
                first_frame_received = asyncio.Queue(maxsize=1)


                #收到影像要做甚麼?
                @pc.on("track")
                async def on_track(track):
                    print(f"[{path}] 接收到 Track: {track.kind}") #列印接收到的 Track 類型
                    if track.kind == "video":
                        try:
                            # 嘗試接收第一幀，設定較長的逾時
                            first_frame = await asyncio.wait_for(track.recv(), timeout=15.0) #接收一幀影像
                            print(f"[{path}] ✅ 成功接收到第一個有效影像幀！")
                            #把第一幀影像放進佇列
                            if not frame_queue.full():
                                frame_queue.put_nowait(first_frame.to_ndarray(format="bgr24"))
                            # 通知主迴圈第一幀已收到
                            await first_frame_received.put(True)

                            # 繼續接收後續的幀
                            while not shutdown_event.is_set():
                                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
                                if not frame_queue.full():
                                    frame_queue.put_nowait(frame.to_ndarray(format="bgr24"))
                        except asyncio.TimeoutError:
                            print(f"[{path}] ⚠️ 接收影像幀超時，將嘗試重新連線。")
                        except Exception as e:
                            print(f"[{path}] ⚠️ Track 接收時發生錯誤: {e}，將嘗試重新連線。")
                        finally:
                            # 如果 on_track 迴圈中斷，確保 first_frame_received 佇列有東西
                            # 以免外層的 await first_frame_received.get() 卡死
                            if first_frame_received.empty():
                                await first_frame_received.put(False)

                # --- 連線流程 ---
                #建立webrtc 連線(開門營業)
                url = f"{mediamtx_base_url}/{path}/whep" #WebRTC 連線 URL
                print(f"[{path}] 正在嘗試連接到 WebRTC: {url} ...")
                pc.addTransceiver("video", direction="recvonly") #只接收影像
                offer = await pc.createOffer() #建立 Offer SDP
                await pc.setLocalDescription(offer) #設定本地描述

                #與伺服器交換連線資訊(確認訂單)
                async with aiohttp.ClientSession() as session: #建立非同步 HTTP Session
                    async with session.post(url, data=pc.localDescription.sdp, #發送 Offer SDP  這時候 mediamtx 伺服器會回傳 Answer SDP 如果可以就會傳track
                                            headers={"Content-Type": "application/sdp"}) as resp:
                        if resp.status != 201:
                            print(f"[{path}] ❌ 連線失敗，狀態碼: {resp.status}。將在 5 秒後重試。")
                            await asyncio.sleep(5)
                            continue  # 跳到下一次 while 迴圈

                        answer_sdp = await resp.text() #取得 Answer SDP(SDP 包含 編解碼器 網路位址等資訊)


                        # 使用 RTCSessionDescription 來封裝 SDP，而不是不存在的 parse_sdp
                        remote_description = RTCSessionDescription(sdp=answer_sdp, type="answer") #建立遠端描述
                        await pc.setRemoteDescription(remote_description) #設定遠端描述

                        print(f"[{path}] ✅ WebRTC SDP 交換成功！等待第一個影像幀...")

                # 等待 on_track 成功接收到第一幀，或逾時
                try:
                    success = await asyncio.wait_for(first_frame_received.get(), timeout=15.0)
                    if not success:
                        print(f"[{path}] ⚠️ on_track 回呼中發生錯誤，準備重連。")
                        continue
                except asyncio.TimeoutError:
                    print(f"[{path}] ❌ 等待第一個影像幀逾時 (15s)，將嘗試重新連線。")
                    continue

                # 如果連線成功且收到幀，就保持連線狀態
                print(f"[{path}] 連線穩定，進入監控狀態...")
                while not shutdown_event.is_set() and pc.connectionState in ["connected", "connecting"]:
                    await asyncio.sleep(1)
                print(f"[{path}] 連線狀態變為 {pc.connectionState}，準備重連。")

            except asyncio.CancelledError:
                print(f"[{path}] 接收任務被取消。")
                break  # 退出 while 迴圈
            except Exception as e:
                print(f"[{path}] ❌ 發生未預期的嚴重錯誤: {e}")
                traceback.print_exc()
            finally:
                if pc:
                    await pc.close()
                if not shutdown_event.is_set():
                    print(f"[{path}] 連線已關閉，將在 5 秒後自動重新連線...")
                    await asyncio.sleep(5)

    loop.run_until_complete(receiver_task())



def webcam_receiver_thread(camera_index, frame_queue, shutdown_event): #本地攝像頭接收執行緒
    print(f"🎥 正在啟動本地攝像頭 #{camera_index} ...") #列印啟動訊息
    cap = cv2.VideoCapture(camera_index) #開啟本地攝像頭
    if not cap.isOpened(): #檢查是否成功開啟攝像頭
        print(f"錯誤：無法開啟攝像頭 #{camera_index}")
        return
    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ 無法從攝像頭 #{camera_index} 讀取影像幀。")
            break
        if not frame_queue.full():
            frame_queue.put_nowait(frame)
        time.sleep(0.01)
    cap.release()
    print(f"📷 攝像頭執行緒 #{camera_index} 已結束。")



# --- 3. 主程式 ---
def main():
    trip_id = f"trip_{uuid.uuid4().hex[:8]}"
    print(f"====== 本次行程 ID: {trip_id} ======")
    notify_backend(RECORDING_STATUS_ENDPOINT, {"session_id": trip_id, "status": "start"})

    shutdown_event = threading.Event() #結束通知

    #1.準備工作
    frame_queues, receiver_threads = {}, [] #緩衝區


    if INPUT_SOURCE_MODE == "WEBRTC":
        active_paths = stream_paths
        print(f"--- 啟動 WebRTC 模式，處理串流: {active_paths} ---")
        for path in active_paths:
            frame_queues[path] = Queue(maxsize=5)
            #可以直接呼叫

            #請一個服務生(啟動Webrtc執行緒)
            thread = threading.Thread(target=webrtc_receiver_thread, args=(path, frame_queues[path], shutdown_event), daemon=True)
            receiver_threads.append(thread); thread.start()
    elif INPUT_SOURCE_MODE == "WEBCAM":
        active_paths = [f"webcam{i}" for i in local_camera_indices]
        print(f"--- 啟動本地 Webcam 模式，處理鏡頭: {local_camera_indices} ---")
        for i, path in zip(local_camera_indices, active_paths):
            frame_queues[path] = Queue(maxsize=5)
            thread = threading.Thread(target=webcam_receiver_thread, args=(i, frame_queues[path], shutdown_event), daemon=True)
            receiver_threads.append(thread); thread.start()
    else:
        print(f"錯誤：未知的 INPUT_SOURCE_MODE: {INPUT_SOURCE_MODE}"); return




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
                        frame_orig = frame_queues[path].get_nowait()  #從緩衝區拿訂單
                    except Empty:
                        continue

                window_name = f"Intelligent Recorder - {path}"
                track_histories = all_track_histories[path]
                track_predictions = all_track_predictions[path]
                frame_idx = all_frame_idx[path]
                cached_seg_map = all_cached_seg_maps[path]

                if path not in video_writers and SAVE_VIDEO:
                    height, width, _ = frame_orig.shape
                    save_path = os.path.join(RECORD_OUTPUT_DIR, f"{path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                    video_writers[path] = {'writer': VideoWriterThread(save_path, (width, height)), 'start_time': time.time(), 'path': save_path, 'danger_zone': np.array([[int(width*0.25), int(height*0.6)], [int(width*0.75), int(height*0.6)], [int(width*0.95), height-1], [int(width*0.05), height-1]], np.int32)}
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
                    img_tensor = torch.from_numpy(img_rgb.astype(np.float32)/255.0).permute(2,0,1)
                    mean, std = torch.tensor([0.485,0.456,0.406],device=device).view(3,1,1), torch.tensor([0.229,0.224,0.225],device=device).view(3,1,1)
                    seg_input_tensor = ((img_tensor.to(device) - mean)/std).unsqueeze(0)
                    with torch.no_grad():
                        seg_logits = seg_model(seg_input_tensor)
                        if isinstance(seg_logits, dict): seg_logits = seg_logits['out']
                        cached_seg_map = torch.argmax(seg_logits.squeeze(), dim=0).cpu().numpy()
                    all_cached_seg_maps[path] = cached_seg_map

                tracks_to_predict = []
                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = obj[:5]
                    track_histories[int(track_id)].append([(x1+x2)/2, y2])
                    if len(track_histories[int(track_id)]) == obs_len:
                        tracks_to_predict.append(int(track_id))

                if tracks_to_predict:
                    with torch.no_grad():
                        num_to_predict = len(tracks_to_predict)
                        batch_hist = torch.from_numpy(np.array([list(track_histories[tid]) for tid in tracks_to_predict])).float().to(device)
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
                            track_predictions[track_id] = best_future[i].cpu().numpy()

                orig_h, orig_w = frame_orig.shape[:2]
                w_scale, h_scale = orig_w/MODEL_INPUT_WIDTH, orig_h/MODEL_INPUT_HEIGHT

                danger_zone_poly = video_writers.get(path, {}).get('danger_zone')
                is_danger = False
                if danger_zone_poly is not None:
                    is_danger = any(cv2.pointPolygonTest(danger_zone_poly, (int(p[0]), int(p[1])), False) >= 0 for tid in track_predictions for p in (track_predictions[tid] * [w_scale, h_scale]))

                    if is_danger and (time.time() - last_danger_notify_time[path] > 10):
                        # 1. 更新時間，等於進入“冷卻倒數”
                        last_danger_notify_time[path] = time.time()

                        # 2. 準備數據並發送通知
                        print(f"[{path}] 偵測到危險！發送通知並冷卻 10 秒。")
                        danger_data = {
                            "trip_id": trip_id,
                            "event_type": "軌跡預測警告",
                            "description": f"鏡頭 [{path}] 偵測到有物體軌跡進入危險區域！"
                        }
                        threading.Thread(target=notify_backend, args=(NOTIFY_DANGER_ENDPOINT, danger_data)).start()


                    zone_color = (0,0,255) if is_danger else (0,255,0)
                    cv2.polylines(canvas, [danger_zone_poly], True, zone_color, 3)
                    # if is_danger:
                    #     text = "!!! WARNING !!!"
                    #     text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 2, 3)
                    #     cv2.putText(canvas, text, ((orig_w-text_size[0])//2, int(orig_h*0.2)), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,255), 3)

                for obj in tracked_objects:
                    x1,y1,x2,y2,tid = [int(p) for p in obj[:5]]
                    x1_o,y1_o,x2_o,y2_o = int(x1*w_scale),int(y1*h_scale),int(x2*w_scale),int(y2*h_scale)
                    cv2.rectangle(canvas, (x1_o,y1_o), (x2_o,y2_o), (0,255,0), 2)
                    cv2.putText(canvas, f"ID:{tid}", (x1_o,y1_o-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    obs_orig_defined = False
                    if tid in track_histories and len(track_histories[tid]) > 1:
                        obs_orig = (np.array(track_histories[tid]) * [w_scale,h_scale]).astype(np.int32)
                        obs_orig_defined = True
                        for k in range(len(obs_orig)-1):
                            cv2.line(canvas, tuple(obs_orig[k]), tuple(obs_orig[k+1]), (255,100,0), 2)
                    if tid in track_predictions and obs_orig_defined:
                        pred_orig = (track_predictions[tid] * [w_scale,h_scale]).astype(int)
                        full_pred = np.vstack([obs_orig[-1], pred_orig])
                        for k in range(len(full_pred)-1):
                            cv2.line(canvas, tuple(full_pred[k]), tuple(full_pred[k+1]), (0,0,255), 2)
                #上菜(顯示結果)
                cv2.imshow(window_name, canvas)
                if path in video_writers:
                    video_writers[path]['writer'].add_frame_to_queue(canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                shutdown_event.set()
                break
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
            relative_path = os.path.join(VIDEO_FILES_DIR, video_filename).replace("\\","/")
            upload_data = { "trip_id": trip_id, "path": path, "relative_path": relative_path, "title": f"PiCam錄影 - {path}", "date": datetime.now().strftime('%m/%d'), "description": f"時長約 {round(time.time() - vw_data['start_time'])} 秒。", }
            notify_backend(UPLOAD_VIDEO_ENDPOINT, upload_data)
        notify_backend(RECORDING_STATUS_ENDPOINT, {"session_id": trip_id, "status": "end"})
        cv2.destroyAllWindows()
        print("程式已結束。")

if __name__ == "__main__":
    main()