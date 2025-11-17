# app.py (真正完整的最終版 - 包含旅行日誌 AI 功能)

import os
import json # 用於處理 JSON 數據 JSON數據就是一種輕量級的數據交換格式 很像python的字典和列表
import uuid  # 用於生成唯一 ID
import requests # 用於發送 HTTP 請求 用於發送 "HTTP" 請求。它簡化了與網絡服務器的交互，使得發送 GET、POST 等請求變得非常容易。
from flask import Flask, request, jsonify, send_from_directory # Flask 框架相關模組 #包含處理請求和回應的功能
from flask_cors import CORS # 用於處理跨域請求 # 用於處理跨域請求的模組。CORS（Cross-Origin Resource Sharing）允許你的後端 API 被來自不同來源（域名、協議或端口）的前端應用程式訪問。
import threading # 用於背景緩存清理
import time  # 用於時間相關操作
import datetime
import firebase_admin # 用於 Firebase Admin SDK
from firebase_admin import credentials, messaging # 用於 Firebase 消息傳遞 # 用於與 Firebase 進行互動，特別是發送推送通知。

from collections import defaultdict # 確保在檔案頂部 import defaultdict

import datetime as dt # 確保頂部 import 的是 dt
from dateutil import parser # 需要安裝 pip install python-dateutil

# 初始化 Firebase Admin SDK
try:
    cred = credentials.Certificate("service-account-key.json") # 載入服務帳戶金鑰檔案
    firebase_admin.initialize_app(cred) # 初始化 Firebase Admin SDK
    print("Firebase Admin SDK 初始化成功！")
except Exception as e:
    print(f"錯誤：Firebase Admin SDK 初始化失敗: {e}")
    print("請確保 serviceAccountKey.json 檔案存在且有效。")





# --- 1. 初始化與配置 ---
current_location = os.path.abspath(os.path.dirname(__file__)) ##當前位置 stream_yolo/real_big_test_TRY_FONRT_END/aaaappppp.py
current_location_webserver_folder = os.path.join(current_location, 'web_server')# aaaappppp這個目錄的資料夾 web_server
app = Flask(__name__, static_folder=current_location_webserver_folder, static_url_path='')  # 指定檔案資料夾
# Flask web_server資料夾
CORS(app)#允許跨域請求  #允許前端程式（JS）呼叫你的 API。



# --- 配置 ---
GROQ_API_KEY = "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" # Groq API 金鑰
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions" # Groq API 端點 這固定的去翻groq的api介紹


video_database= 'videos_db.json' # 影片數據庫文件 這在當前目錄
VIDEO_FILES_DIR = 'videos' # 影片儲存目錄 這在當前目錄的web_server/videos




#傳送 影響
def send_firebase_notification(token, title, body): #目標設備、通知標題、通知內容(body)
    """
    透過 FCM 發送通知到單一設備。
    """
    print("\n--- 準備發送 Firebase 通知 ---")
    print(f"目標 Token: ...{token[-10:]}")
    print(f"標題: {title}")
    print(f"內容: {body}")
    try:
        #建立FCM 訊息物件
        message = messaging.Message(
            #設定通知內容
            notification=messaging.Notification(title=title, body=body),
            token=token, #目標設備 token
            android=messaging.AndroidConfig(
                priority='high',
                # 確保這個 channel_id 與您 Android App 中創建的一致 就是後面那個danger_alert_channel
                notification=messaging.AndroidNotification(channel_id='danger_alert_channel')
            )
        )
        #發送通知 並取得回應
        response = messaging.send(message)
        print(f"成功發送訊息！ Message ID: {response}")
        return True # 代表成功
    except Exception as e:
        print(f"發送訊息時發生錯誤: {e}")
        return False # 代表失敗





# 讀取影片數據庫_函式
def load_videos_from_db():
    video_database_path = os.path.join(current_location, video_database) #join拼接路徑 當前位置+影片數據庫文件 就變成指定路徑到videos_db.json
    if not os.path.exists(video_database_path):
        return [] # 如果文件不存在，返回空列表
    try:
        with open(video_database_path, 'r', encoding='utf-8') as f: #read模式打開 # 讀取數據庫文件
            return json.load(f) # 解析 JSON 並返回 f是文件對象
    except (json.JSONDecodeError, FileNotFoundError):
        return []
##這就代表   當前位置>>>拼接>>>影片數據庫文件 >>>讀取videos_db.json >>>解析json並返回
## 為什麼要解析json並返回?? -> 因為存的數據是json格式的 ->把json格式的數據轉換成python能操作的數據結構(列表/字典) >>>這樣你才能操作這些數據
#至於為什麼都要用try except 因為你不確定這個文件會不會損壞 或是不存在 這樣程式才不會"崩潰"




# 儲存影片數據庫(這就有牽扯到 "write" 了)_函式
def save_videos_to_db(videos):
    video_database_path = os.path.join(current_location, video_database) #一樣用join拼接路徑 當前位置+影片數據庫文件 就變成指定路徑到videos_db.json
    try:
        with open(video_database_path, 'w', encoding='utf-8') as f: # write模式打開
            json.dump(videos, f, ensure_ascii=False, indent=4) #dump是寫入的意思 # 將影片數據寫入 JSON 文件
            # ensure_ascii=False 是為了正確處理非 ASCII 字符（如中文） # indent=4 是為了讓 JSON 文件更易讀
        return True # 成功返回 True 確認寫入成功
    except Exception as e:
        print(f"保存影片數據失敗: {e}")
        return False



#轉換成瀏覽器能夠直接看懂並顯示的 HTML 程式碼

#使用時機有兩個 (1.載入index.html 剛進去前端的時候  2.載入video_detail.html 這就是點進影片的時候)

def generate_video_content_html(video_data, is_detail_page=False): #video_data是影片數據 這也是你自己定義的  is_detail_page是布林值(是否為詳細頁面) #is_detail_page自己定義的參數>>>用來區分影片是在列表頁面還是詳細頁面顯示(就是第二頁html拉)
    content_type = video_data.get("content_type")
    # 影片類型 (本地影片 之類的)   影片類型是你自己定義的參數 # 這是從影片數據中獲取的 在add_video_api裡面有定義
    poster = video_data.get("poster", "") # 影片封面 # 影片封面URL
    width, height = (video_data.get("width", 260), video_data.get("height", 145)) if not is_detail_page else ("", "") #這是排版 # 進去的時候 "我的影片" 那邊每部影片寬高都一樣 #如果是第二頁html就不限制了(他會自適應寬高 去符合手機電腦的頁面)
    width_attr, height_attr = (f'width="{width}"', f'height="{height}"') if not is_detail_page else ("", "")# 影片寬高屬性

    if content_type == "local_video": # 本地影片
        path = video_data.get("path", "") # 影片路徑
        src_path = f"/{path}" # 影片來源路徑 (相對於靜態文件夾)
        return f'<video {width_attr} {height_attr} controls poster="{poster}"><source src="{src_path}" type="video/mp4">您的瀏覽器不支援。</video>'# 影片標籤
        #



# --- 3. 數據緩衝與管理 (保持不變) ---
received_data_buffer = {}
ACTIVE_SESSION_ID = None # 當前活躍的錄影 Session ID
trip_to_token_map = {}



# 定期清理緩衝區，刪除不活躍且過期的數據
def cleanup_buffer():
    current_time = time.time() # 當前時間戳
    keys_to_delete = [] # 要刪除的鍵列表
    for session_id, data_bundle in received_data_buffer.items(): # 遍歷receive_data_buffer 緩衝區 的每個 session_id 和對應的數據
        last_received = data_bundle.get('last_received_time', 0) # 最後接收時間
        if not data_bundle.get('is_active', False) and (current_time - last_received > 900): # 不活躍且超過 15 分鐘
            keys_to_delete.append(session_id) # 標記為刪除
    for session_id in keys_to_delete: # 刪除標記的鍵
        print(f"Cleaning up expired buffer for session: {session_id}") # 日誌輸出
        del received_data_buffer[session_id] # 刪除緩衝區中的數據



# 啟動背景線程定期清理緩衝區
def start_buffer_cleanup_thread():
    thread = threading.Thread(
        target=lambda: (lambda: time.sleep(300) or cleanup_buffer())() or start_buffer_cleanup_thread()) # 每 5 分鐘清理一次
    thread.daemon = True  # 這樣主程序退出時，線程會自動結束
    thread.start() # 啟動線程








# 核心 API 端點
@app.route('/') #'/' 指得是首頁
def index():
    return send_from_directory(app.static_folder, 'index.html') # 這是首頁 這是給前端用的

@app.route('/live_view.html')
def live_view():
    return send_from_directory(app.static_folder, 'live_view.html')

# ( recording_status, get_current_recording_session_id, receive_gps_data, receive_danger_events )



#這是錄影狀態的api #錄影程式會呼叫這個api來告訴後端目前錄影狀態
@app.route('/recording_status', methods=['POST']) # 這是錄影狀態的api #錄影程式會呼叫這個api來告訴後端目前錄影狀態 POST意思就是錄影程式會傳送數據給這個api
def recording_status():
    global ACTIVE_SESSION_ID # 使用全局變數  #名稱叫做 ACTIVE_SESSION_ID (trip_id)
    data = request.json # 取得請求中的 JSON 數據  #這是錄影程式傳來的數據
    session_id = data.get("session_id") # 從錄影程式給的 data 拿id
    status = data.get("status") # 取得狀態 (start 或 end)

    if not session_id or not status:
        return jsonify({"error": "Missing session_id or status"}), 400

    if status == "start":  #如果狀態是 start
        ACTIVE_SESSION_ID = session_id # 設置為當前活躍 Session ID
        if session_id not in received_data_buffer: # 如果 session_id 不在緩衝區中
            received_data_buffer[session_id] = {"gps_data": [], "danger_events": [], "is_active": True,
                                                "last_received_time": time.time()} #在 received_data_buffer 裡建立一個新的 session 資料（包含 GPS、危險事件、活躍狀態、最後接收時間）。
    elif status == "end": # 如果狀態是 end
        if ACTIVE_SESSION_ID == session_id: # 如果是當前活躍的 Session ID
            ACTIVE_SESSION_ID = None # 清除活躍 Session ID
        if session_id in received_data_buffer:
            received_data_buffer[session_id]['is_active'] = False

    return jsonify({"message": "Status updated"}), 200



#這手機端的危險通知api
@app.route('/notify_danger', methods=['POST'])
def notify_danger_api():
    data = request.json
    trip_id = data.get("trip_id")
    description = data.get("description", "偵測到未知危險")

    if trip_id and trip_id in received_data_buffer:
        # 我们需要一个时间戳来匹配 GPS，这里我们从 AI 客户端获取
        event_timestamp = data.get('timestamp')  # 获取 AI 客户端传来的 ISO 8601 时间戳

        if event_timestamp:  # 确保时间戳存在
            new_event = {
                "description": description,
                "timestamp": event_timestamp
            }
            received_data_buffer[trip_id]['danger_events'].append(new_event)
        print(f"✅ 已為行程 [{trip_id}] 記錄一筆危險事件。")

    if not trip_id:
        return jsonify({"error": "Missing trip_id"}), 400

    print(f"收到來自 [{trip_id}] 的危險警報: {description}")

    # 【【【 核心修改 】】】
    # 直接在這裡定義你的固定手機 Token
    # 不再需要  查詢 trip_to_token_map
    FIXED_DEVICE_TOKEN = "dyvMb1dHTFu5m5HQxDILcw:APA91bFfC_v08H83Qea9dFEWXFpam68fIAFhbpfiJGin5p0UOx-iQQBOcuTBzWV5UxJLIlPP6rVu-xQI_mZlcfF5wQjdEGLpEdTsxVC4IQy-HJgyXvE_S2E"

    if not FIXED_DEVICE_TOKEN:
        print(f"錯誤：後端沒有設定固定的設備 Token。")
        return jsonify({"error": "Backend server is not configured with a device token"}), 500

    # 直接使用這個固定的 Token 來發送通知
    success = send_firebase_notification(
        token=FIXED_DEVICE_TOKEN,
        title='⚠️ 行車安全警報 ⚠️',
        body=description
    )

    # 根據發送結果，回傳不同的狀態
    if success:
        return jsonify({"status": "success", "message": "Notification sent to fixed device"}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to send notification"}), 500





#把id 放這裡 android 來訪 拿ACTIVE_SESSION_ID
@app.route('/get_current_recording_session_id', methods=['GET'])  # 這是給手機App查詢目前錄影的session_id #GET意思就是手機會來這個api拿資料
def get_current_recording_session_id():
    return jsonify({"session_id": ACTIVE_SESSION_ID})  # 回傳目前的 Session ID 格式是 {"session_id": "session_12345678"} 或 {"session_id": null}
#jsonify 是 Flask 提供的一個函數，用來將 Python 字典或列表轉換成 JSON 格式的 並回傳給Android
#Android 端收到的 JSON 格式數據後，可以使用相應的 JSON 解析庫來提取 session_id 的值。
#例如，如果 ACTIVE_SESSION_ID 是 "session_12345678"，那麼回傳的 JSON 會是 {"session_id": "session_12345678"}。
#如果沒有活躍的錄影會話，ACTIVE_SESSION_ID 會是 None，回傳的 JSON 會是 {"session_id": null}。
#這樣手機App就能知道目前錄影的 session_id 是多少，然後在傳送 GPS 和危險事件數據時使用這個 session_id。





@app.route('/receive_gps_data', methods=['POST']) # 這是接收GPS數據的api POST意思就是手機會傳送數據給這個api
def receive_gps_data():
    data = request.json # 取得請求中的 JSON 數據 這個JSON數據是手機傳來的
    session_id = data.get("session_id")  #這邊一樣要用data.get 從android 用okhttp 那邊POST 數據

    if not session_id or data.get("latitude") is None: #如果session id 是空的 或是android data那邊的數據傳過來的經緯度座標是空的就:
        return jsonify({"error": "Missing GPS data fields"}), 400 #error
    if session_id not in received_data_buffer: #如果session_id 不在 received_data_buffer
        if ACTIVE_SESSION_ID == session_id:
            received_data_buffer[session_id] = {"gps_data": [], "danger_events": [], "is_active": True,
                                                "last_received_time": time.time()}
            #在 received_data_buffer 裡建立一個新的 session 資料（包含 GPS、危險事件、活躍狀態、最後接收時間）。
        else:
            return jsonify({"message": "GPS for inactive session"}), 202
    received_data_buffer[session_id]["gps_data"].append(
        {"lat": data["latitude"], "lng": data["longitude"], "timestamp": data["timestamp"],
         "accuracy": data["accuracy"]})
    return jsonify({"message": "GPS received"}), 200







#這是上傳錄製影片的api # 這是給錄影程式在錄影結束後會呼叫這個api來上傳影片資訊
@app.route('/upload_recorded_video', methods=['POST'])

def upload_recorded_video():

    data = request.json # 取得請求中的 JSON 數據 這個JSON數據是錄影程式傳來的
    #後端這邊取得資料->代表錄影程式利用requests.post這個方法把資料傳給後端(http://你的伺服器ip:5000/upload_recorded_video, json=data) 這個upload_data就是data
    #假設data長這樣:

    # data={
    #       "session_id": "session_12345678",
    #       "relative_path": "videos/vid_cam_abcdef.mp4",
    #       "title": "Webcam錄影 - 10/05 14:
    # }
    #錄影程式就會長這樣 requests.post("http://你的伺服器ip:5000/upload_recorded_video", json=data)  <-----這個data就是上面那個data 並且json=data代表把data轉成json格式傳給後端
    #這樣後端才能用 data=request.json 取得這些數據


    videos = load_videos_from_db()  # 加載現有影片數據庫 這是讀取你本地的影片數據庫
    trip_id = data.get("trip_id") #從讀取到的數據庫 拿id
    session_id = data.get("session_id")
    path_name = data.get("path", "cam")
    if not trip_id and not session_id: return jsonify({"error": "Missing trip_id or session_id"}), 400
    data_key = trip_id if trip_id else session_id # 優先使用 trip_id，否則使用 session_id
    gps_trace = received_data_buffer.get(data_key, {}).get('gps_data', []) # 從緩衝區獲取 GPS 數據
    danger_events = received_data_buffer.get(data_key, {}).get('danger_events', []) # 從緩衝區獲取危險事件
    video_id = f"vid_{path_name}_{uuid.uuid4().hex[:6]}" # 生成唯一影片 ID 這就不是trip_id了!!!   這是給影片自己用的id
    new_video = {"id": video_id, "trip_id": trip_id, "date": data.get("date"), "title": data.get("title"),
                 "description": data.get("description"), "content_type": "local_video",
                 "path": data.get("relative_path"), "gps_trace": gps_trace, "danger_events": danger_events,
                 "location": gps_trace[0] if gps_trace else {}} # 建立新影片數據
    videos.append(new_video) # 添加到影片列表
    if save_videos_to_db(videos): return jsonify({"message": "Video uploaded", "video_id": new_video['id']}), 201
    return jsonify({"error": "Failed to save video"}), 500




#---- 手動 ----
@app.route('/add_video', methods=['POST']) #這個是新增影片的api # 這是給前端上傳影片資訊用的api
def add_video_api():
    data = request.json # 取得請求中的 JSON 數據 # 這是前端傳來的數據 #這在前端的upload_video.js裡面
    if not data or not data.get('title') or not data.get('content_type'): return jsonify(
        {"error": "缺少標題或影片類型"}), 400 # 檢查必要字段 # 這是檢查前端傳來的數據是否有標題和影片類型


    videos = load_videos_from_db() # 加載現有影片數據庫 # 這是讀取你本地的影片數據庫

    gps_trace, danger_events = [], [] # 預設為空列表


    new_video = {"id": f"vid_manual_{uuid.uuid4().hex[:6]}", "trip_id": None, "date": data.get("date", "未知日期"),
                 "title": data.get("title"), "description": data.get("description", ""),
                 "content_type": data.get("content_type"), "path": data.get("path", ""),
                 "gps_trace": gps_trace, "danger_events": danger_events, "width": data.get("width", 260),
                 "height": data.get("height", 145), "location": data.get("location", {})} # 建立新影片數據
    videos.append(new_video) # 添加到影片列表
    if save_videos_to_db(videos): # 儲存到數據庫
        new_video['content'] = generate_video_content_html(new_video) # 生成影片內容 HTML
        return jsonify(new_video), 201
    return jsonify({"error": "儲存影片失敗"}), 500



# 影片管理 API
@app.route('/get_videos', methods=['GET'])
def get_videos_api():
    videos_db = load_videos_from_db() # 加載現有影片數據庫 (從 videos_db.json 檔案中讀取所有影片的原始數據)
    videos_for_frontend = [dict(v, content=generate_video_content_html(v)) for v in videos_db] # 生成每個影片的內容 HTML #
    return jsonify(videos_for_frontend) # 回傳影片列表




#根據前端提供的「影片 ID」，去資料庫裡找出那「唯一」的一部影片的詳細資料，交給詳情頁   (就是第二頁拉)
@app.route('/get_video/<video_id>', methods=['GET'])
def get_single_video_api(video_id): # 獲取單一影片詳情 API
    found_video = next((v for v in load_videos_from_db() if v.get('id') == video_id), None) # 查找指定 ID 的影片
    if found_video: # 如果找到影片
        return jsonify(dict(found_video, content=generate_video_content_html(found_video, is_detail_page=True))) # 生成詳細頁面的內容 HTML
    return jsonify({"error": "找不到該影片"}), 404





# 刪除影片 API
@app.route('/delete_video/<video_id>', methods=['DELETE'])
def delete_video_api(video_id):
    # (您原有的 delete_video_api)
    videos = load_videos_from_db()
    videos_to_keep = [v for v in videos if v.get('id') != video_id]
    if len(videos_to_keep) < len(videos):
        save_videos_to_db(videos_to_keep)
        return jsonify({"message": f"影片 ID {video_id} 已刪除"}), 200
    return jsonify({"error": f"找不到影片 ID {video_id}"}), 404






# 【新增】輔助函式：將經緯度轉換成地名
def get_location_name(lat, lng):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&zoom=18&addressdetails=1" # 使用 OpenStreetMap 的 Nominatim 服務
        headers = {'User-Agent': 'MyCoolMotorcycleApp/1.0'} # Nominatim 要求提供 User-Agent
        response = requests.get(url, headers=headers, timeout=5) # 發送請求
        response.raise_for_status() # 檢查請求是否成功
        data = response.json() # 解析 JSON 回應
        address = data.get('address', {}) # 獲取地址部分
        road = address.get('road', '') # 街道名稱
        suburb = address.get('suburb', address.get('city_district', '')) # 區域名稱
        poi = address.get('amenity', address.get('shop', address.get('tourism', ''))) # 重要地點名稱

        # 返回一個更詳細的字典，而不是單一字串
        location_info = {"poi": poi, "road": road, "suburb": suburb} # 重要地點、街道、區域
        print(f"成功獲取地點資訊: {location_info}")
        return location_info
    except Exception as e:
        print(f"無法獲取地點名稱: {e}")
        return None  # 失敗時返回 None






# AI 報告 API
def find_closest_gps_point(event_timestamp_str, gps_trace):
    """
    为给定的 UTC 事件时间戳，寻找与“伪装成 UTC 的本地时间”GPS 轨迹中最接近的点。
    """
    if not gps_trace or not event_timestamp_str:
        return None
    try:
        # 1. 解析来自 AI 程式的【真实 UTC 时间戳】
        event_dt_utc = parser.isoparse(event_timestamp_str)

        # 【【【 关键修正 】】】
        # 将其转换为台湾本地时间 (UTC+8)，以匹配 Android 端发送的“伪 UTC 时间”
        # 我们创建一个 timedelta 物件来表示 8 小时的时差
        taiwan_tz_offset = dt.timedelta(hours=8)
        event_dt_local = event_dt_utc + taiwan_tz_offset

        min_diff = float('inf')
        closest_point = None

        for point in gps_trace:
            # 2. 解析来自 Android App 的【伪 UTC 时间戳】
            # dateutil.parser 会把它当作一个不含时区资讯的“纯真”时间来解析
            gps_dt_pseudo_utc = parser.isoparse(point['timestamp'])

            # 3. 现在，event_dt_local 和 gps_dt_pseudo_utc 都在“台湾本地时间”的基准上了
            #    虽然它们可能都没有正确的时区资讯，但它们的数值是可以直接比较的
            diff = abs((event_dt_local - gps_dt_pseudo_utc).total_seconds())

            if diff < min_diff:
                min_diff = diff
                closest_point = point

        if closest_point and min_diff < 15:
            print(f"✅ 成功匹配事件與 GPS！最小時間差: {min_diff:.2f} 秒")
            return closest_point
        else:
            if closest_point:
                # 这里的 min_diff 应该会变小，但如果还是很大，可能还有其他问题
                print(f"⚠️ 匹配失敗：找到了最近的 GPS 點，但時間差仍然过大 ({min_diff:.2f} 秒 > 15 秒)。")
            else:
                print(f"⚠️ 匹配失敗：GPS 軌跡中沒有任何時間點。")

    except Exception as e:
        print(f"❌ 解析時間戳時發生嚴重錯誤: {e}. Event Timestamp: '{event_timestamp_str}'")

    return None
###################################
@app.route('/get_groq_ai_response', methods=['POST'])
def get_groq_ai_api():
    if not GROQ_API_KEY or "你的Groq_API_Key" in GROQ_API_KEY:
        return jsonify({"error": "AI 服務未配置 API 金鑰"}), 500

    data = request.json
    danger_events = data.get('danger_events', [])
    gps_trace = data.get('gps_trace', [])

    prompt = "你是一位專業的行車安全助理與旅行日誌作家。你的任務是根據我提供的數據，嚴格按照指示生成一份精簡的報告，包含『行車影片總結』和『安全建議』兩個部分。不要包含日期、駕駛人等多餘欄位。用詞客觀，語氣專業。\n\n"

    # --- 核心修改 ---
    if danger_events and gps_trace:  # 只有在同時有危險事件和 GPS 數據時，才進行定位匯總
        prompt += "這是一次包含危險事件的行程。數據分析如下：\n"

        event_locations = defaultdict(int)
        unlocated_events_count = 0

        for event in danger_events:
            # 假設危險事件的描述格式為 "鏡頭 [cam0] 偵測到..."，時間戳為 'HH:MM:SS'
            event_timestamp = event.get('timestamp')

            closest_gps = find_closest_gps_point(event_timestamp, gps_trace)

            if closest_gps:
                location_info = get_location_name(closest_gps['lat'], closest_gps['lng'])
                if location_info and (location_info.get('road') or location_info.get('suburb')):
                    location_name = location_info.get('road') or location_info.get('suburb')
                    event_locations[location_name] += 1
                else:
                    unlocated_events_count += 1
            else:
                unlocated_events_count += 1

        if event_locations:
            prompt += "[危險事件地理位置匯總]:\n"
            for location, count in event_locations.items():
                prompt += f"- 在 **{location}** 附近，共偵測到 **{count}** 次潛在危險。\n"

        if unlocated_events_count > 0:
            prompt += f"- 另有 **{unlocated_events_count}** 次潛在危險無法精確匹配地理位置。\n"

        prompt += "\n請在『行車影片總結』中，提及上述發生危險事件的地點。然後在『安全建議』中，提出針對性的改進建議，例如『行經崇義街時應特別注意巷口來車』。\n"

    elif gps_trace and len(gps_trace) > 1:  # 如果沒有危險事件，但有 GPS 軌跡
        # 這部分的程式碼完全複製您原有的版本，保持不變
        start_point = gps_trace[0]
        end_point = gps_trace[-1]
        waypoints = []
        if len(gps_trace) > 5:
            mid_point_1 = gps_trace[len(gps_trace) // 4]
            mid_point_2 = gps_trace[(len(gps_trace) * 3) // 4]
            waypoints.append(get_location_name(mid_point_1['lat'], mid_point_1['lng']))
            waypoints.append(get_location_name(mid_point_2['lat'], mid_point_2['lng']))
        start_info = get_location_name(start_point['lat'], start_point['lng'])
        end_info = get_location_name(end_point['lat'], end_point['lng'])
        start_desc = f"座標 [{start_point['lat']:.4f}, {start_point['lng']:.4f}]"
        if start_info and (start_info.get('poi') or start_info.get('road')):
            start_desc = f"{start_info.get('suburb', '')} 的 {start_info.get('poi') or start_info.get('road')}"
        end_desc = f"座標 [{end_point['lat']:.4f}, {end_point['lng']:.4f}]"
        if end_info and (end_info.get('poi') or end_info.get('road')):
            end_desc = f"{end_info.get('suburb', '')} 的 {end_info.get('poi') or end_info.get('road')}"
        prompt += "這是一次安全的行程。數據如下：\n"
        prompt += f"[行程路線分析]:\n"
        prompt += f"- 起點: {start_desc}\n"
        if waypoints:
            waypoint_descs = [f"{info.get('suburb', '')} 的 {info.get('poi') or info.get('road')}" for info in waypoints
                              if info]
            if waypoint_descs:
                prompt += f"- 途經: {', '.join(waypoint_descs)}\n"
        prompt += f"- 終點: {end_desc}\n\n"
        prompt += "請在『行車影片總結』中，精準地描述這次從起點出發，經過途經點，最終抵達終點的旅程。然後在『安全建議』中提供通用且簡易的建議，條列式最多2項即可。\n"
    else:  # 如果連 GPS 數據都沒有
        # 這部分的程式碼也複製您原有的版本，保持不變
        prompt += "數據不足，無法生成詳細報告。\n\n請在『行車影片總結』中說明本次行程缺乏足夠的 GPS 或事件數據。並在『安全建議』中提供通用行車安全建議，條列式最多2項即可。\n"

    # 後續的 Groq API 請求邏輯完全不變
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.4}  # 建議使用 llama3-8b
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        ai_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "AI 未能提供有效回應。")
        return jsonify({'aiResponse': ai_text.strip()})
    except requests.exceptions.RequestException as e:
        print(f"Groq API 請求錯誤: {e}")
        # 在開發階段，返回詳細錯誤給前端，方便除錯
        return jsonify({"error": f"請求 AI 服務失敗: {str(e)}"}), 503


# --- 6. 啟動器 ---
if __name__ == '__main__':
    if not os.path.exists(video_database):
        save_videos_to_db([])
    print(f"Flask App Root Path: {app.root_path}")
    print(f"Flask Static Folder: {app.static_folder}")
    start_buffer_cleanup_thread()
    app.run(debug=True, host="0.0.0.0", port=5000)