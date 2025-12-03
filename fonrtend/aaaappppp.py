# app.py (AI 修復版 - 升級 Llama 3.3 模型 + 詳細錯誤顯示)

import os
import json
import uuid
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading
import time
import datetime
import firebase_admin
from firebase_admin import credentials, messaging
from collections import defaultdict
import datetime as dt
from dateutil import parser

# --- 1. 初始化與配置 ---
current_location = os.path.abspath(os.path.dirname(__file__))
current_location_webserver_folder = os.path.join(current_location, 'web_server')
app = Flask(__name__, static_folder=current_location_webserver_folder, static_url_path='')
CORS(app)

# 初始化 Firebase
try:
    cred = credentials.Certificate("service-account-key.json")
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK 初始化成功！")
except Exception as e:
    print(f"⚠️ Firebase 初始化跳過 (若不需要App通知可忽略): {e}")

# --- 配置 (請填寫您的 Key) ---
# ▼▼▼ 請填入您的 Groq API Key ▼▼▼
GROQ_API_KEY = "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" # Groq API 金鑰


# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
video_database = 'videos_db.json'


# --- 2. 資料庫與輔助函式 ---

def send_firebase_notification(token, title, body):
    try:
        if not token: return False
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=token,
            android=messaging.AndroidConfig(
                priority='high',
                notification=messaging.AndroidNotification(channel_id='danger_alert_channel')
            )
        )
        response = messaging.send(message)
        return True
    except Exception as e:
        print(f"FCM 發送失敗: {e}")
        return False


def load_videos_from_db():
    video_database_path = os.path.join(current_location, video_database)
    if not os.path.exists(video_database_path): return []
    try:
        with open(video_database_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def save_videos_to_db(videos):
    video_database_path = os.path.join(current_location, video_database)
    try:
        with open(video_database_path, 'w', encoding='utf-8') as f:
            json.dump(videos, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"保存失敗: {e}")
        return False


def generate_video_content_html(video_data, is_detail_page=False):
    content_type = video_data.get("content_type")
    poster = video_data.get("poster", "")
    width = video_data.get("width", 260)
    height = video_data.get("height", 145)
    width_attr = f'width="{width}"' if not is_detail_page else 'style="width:100%"'
    height_attr = f'height="{height}"' if not is_detail_page else ''

    if content_type == "local_video":
        path = video_data.get("path", "")
        src_path = f"/{path}"
        return f'<video {width_attr} {height_attr} controls poster="{poster}"><source src="{src_path}" type="video/mp4">不支援。</video>'
    return ""


# --- 3. 數據緩衝與管理 ---
received_data_buffer = {}
ACTIVE_SESSION_ID = None


def cleanup_buffer():
    current_time = time.time()
    keys_to_delete = []
    for session_id, data_bundle in received_data_buffer.items():
        last_received = data_bundle.get('last_received_time', 0)
        # 若不活躍且超過 15 分鐘沒更新，則刪除
        if not data_bundle.get('is_active', False) and (current_time - last_received > 900):
            keys_to_delete.append(session_id)
    for session_id in keys_to_delete:
        print(f"清理過期 Session: {session_id}")
        del received_data_buffer[session_id]


def start_buffer_cleanup_thread():
    thread = threading.Thread(
        target=lambda: (lambda: time.sleep(300) or cleanup_buffer())() or start_buffer_cleanup_thread())
    thread.daemon = True
    thread.start()


# --- 4. 路由與 API ---

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/live_view.html')
def live_view():
    return send_from_directory(app.static_folder, 'live_view.html')


# === 【核心新增】儀表板數據 API ===
@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    # 計算總影片數
    videos_count = len(load_videos_from_db())

    # 獲取當前 Session 的危險警報
    current_alerts = []
    status = "online"  # 預設為待機 (online)

    if ACTIVE_SESSION_ID and ACTIVE_SESSION_ID in received_data_buffer:
        status = "recording"  # 錄影中
        # 取得該 Session 的危險事件，並倒序排列（最新的在前面）
        events = received_data_buffer[ACTIVE_SESSION_ID].get('danger_events', [])
        current_alerts = sorted(events, key=lambda x: x.get('timestamp', ''), reverse=True)

    return jsonify({
        "status": status,
        "session_id": ACTIVE_SESSION_ID,
        "alerts": current_alerts,
        "total_videos": videos_count,
        "active_cameras": 2
    })


# === 錄影狀態更新 ===
@app.route('/recording_status', methods=['POST'])
def recording_status():
    global ACTIVE_SESSION_ID
    data = request.json
    session_id = data.get("session_id")
    status = data.get("status")

    if not session_id or not status:
        return jsonify({"error": "Missing data"}), 400

    if status == "start":
        print(f"🔔 收到錄影開始通知: {session_id}")  # Log
        ACTIVE_SESSION_ID = session_id
        if session_id not in received_data_buffer:
            received_data_buffer[session_id] = {
                "gps_data": [], "danger_events": [], "is_active": True,
                "last_received_time": time.time()
            }
    elif status == "end":
        print(f"🔕 收到錄影結束通知: {session_id}")  # Log
        if ACTIVE_SESSION_ID == session_id:
            ACTIVE_SESSION_ID = None
        if session_id in received_data_buffer:
            received_data_buffer[session_id]['is_active'] = False

    return jsonify({"message": "Status updated"}), 200


# === 危險通知 ===
@app.route('/notify_danger', methods=['POST'])
def notify_danger_api():
    data = request.json
    trip_id = data.get("trip_id")
    description = data.get("description", "偵測到未知危險")
    timestamp = data.get("timestamp", datetime.datetime.utcnow().isoformat())

    # 存入記憶體
    if trip_id and trip_id in received_data_buffer:
        new_event = {
            "description": description,
            "timestamp": timestamp,
            "id": uuid.uuid4().hex[:8]  # 給前端用的唯一 ID
        }
        received_data_buffer[trip_id]['danger_events'].append(new_event)
        received_data_buffer[trip_id]['last_received_time'] = time.time()
        print(f"✅ 記錄危險: {description}")

    # 發送 Firebase
    # ▼▼▼ 請填入您手機的 Token ▼▼▼
    FIXED_DEVICE_TOKEN = "eDalQlkbT9CNPks0eUSPRx:APA91bF6yuc2xUv3sWRDvuPNCNk4kUatCNGOImCSGStYJwFk29LlLIzzMEv77Rg-4HKJDurmCH6suylp3CtANl8hu9NiTYi3z7E2kcNR4SxEBCj0YCZgaJA"

    if FIXED_DEVICE_TOKEN:
        send_firebase_notification(FIXED_DEVICE_TOKEN, '⚠️ 行車安全警報 ⚠️', description)

    return jsonify({"status": "received"}), 200


@app.route('/get_current_recording_session_id', methods=['GET'])
def get_current_recording_session_id():
    return jsonify({"session_id": ACTIVE_SESSION_ID})


@app.route('/receive_gps_data', methods=['POST'])
def receive_gps_data():
    data = request.json
    session_id = data.get("session_id")
    if not session_id or 'latitude' not in data: return jsonify({"error": "Missing GPS"}), 400

    if session_id not in received_data_buffer:
        received_data_buffer[session_id] = {"gps_data": [], "danger_events": [], "is_active": True,
                                            "last_received_time": time.time()}

    received_data_buffer[session_id]["gps_data"].append({
        "lat": data["latitude"], "lng": data["longitude"],
        "timestamp": data["timestamp"], "accuracy": data["accuracy"]
    })
    received_data_buffer[session_id]['last_received_time'] = time.time()
    return jsonify({"message": "GPS received"}), 200


@app.route('/upload_recorded_video', methods=['POST'])
def upload_recorded_video():
    data = request.json
    videos = load_videos_from_db()
    trip_id = data.get("trip_id") or data.get("session_id")

    # 嘗試從 buffer 撈資料
    buffer_data = received_data_buffer.get(trip_id, {})
    gps_trace = buffer_data.get('gps_data', [])
    danger_events = buffer_data.get('danger_events', [])

    new_video = {
        "id": f"vid_{uuid.uuid4().hex[:6]}",
        "trip_id": trip_id,
        "date": data.get("date"),
        "title": data.get("title"),
        "description": data.get("description"),
        "content_type": "local_video",
        "path": data.get("relative_path"),
        "gps_trace": gps_trace,
        "danger_events": danger_events,
        "location": gps_trace[0] if gps_trace else {}
    }
    videos.append(new_video)
    if save_videos_to_db(videos):
        return jsonify({"message": "Uploaded", "id": new_video['id']}), 201
    return jsonify({"error": "Save failed"}), 500


@app.route('/add_video', methods=['POST'])
def add_video_api():
    data = request.json
    videos = load_videos_from_db()

    # 【修改重點】: 支援從前端接收 Demo 數據
    new_video = {
        "id": f"vid_manual_{uuid.uuid4().hex[:6]}",
        "trip_id": None,
        "date": data.get("date", "未知"),
        "title": data.get("title"),
        "description": data.get("description", ""),
        "content_type": data.get("content_type"),
        "path": data.get("path"),

        # ▼ 允許接收前端測試數據 ▼
        "gps_trace": data.get("gps_trace", []),
        "danger_events": data.get("danger_events", []),
        # ▲ 允許接收前端測試數據 ▲

        "width": 260, "height": 145,
        "location": data.get("location", {})
    }
    videos.append(new_video)
    if save_videos_to_db(videos):
        new_video['content'] = generate_video_content_html(new_video)
        return jsonify(new_video), 201
    return jsonify({"error": "Failed"}), 500

#########################################################
# --- 新增：處理真實檔案上傳的 API ---
@app.route('/upload_video_file', methods=['POST'])
def upload_video_file_api():
    if 'video_file' not in request.files:
        return jsonify({"error": "沒有收到檔案"}), 400

    file = request.files['video_file']
    if file.filename == '':
        return jsonify({"error": "未選擇檔案"}), 400

    if file:
        # 1. 確保檔名安全 (避免 ../ 這種攻擊)
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)

        # 2. 定義儲存路徑 (存到 web_server/videos/)
        save_folder = os.path.join(current_location_webserver_folder, 'videos')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, filename)

        # 3. 儲存檔案
        try:
            file.save(save_path)
            # 回傳相對路徑給前端
            return jsonify({
                "message": "上傳成功",
                "path": f"videos/{filename}",
                "filename": filename
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

##############################################################
@app.route('/get_videos', methods=['GET'])
def get_videos_api():
    db = load_videos_from_db()
    # 倒序排列，新的在前面
    db.reverse()
    for v in db:
        v['content'] = generate_video_content_html(v)
    return jsonify(db)


@app.route('/get_video/<video_id>', methods=['GET'])
def get_single_video_api(video_id):
    video = next((v for v in load_videos_from_db() if v.get('id') == video_id), None)
    if video:
        v_copy = video.copy()
        v_copy['content'] = generate_video_content_html(v_copy, is_detail_page=True)
        return jsonify(v_copy)
    return jsonify({"error": "Not found"}), 404


@app.route('/delete_video/<video_id>', methods=['DELETE'])
def delete_video_api(video_id):
    videos = load_videos_from_db()
    new_list = [v for v in videos if v.get('id') != video_id]
    if len(new_list) < len(videos):
        save_videos_to_db(new_list)
        return jsonify({"message": "Deleted"}), 200
    return jsonify({"error": "Not found"}), 404


# --- AI 與 地點 功能 ---
def get_location_name(lat, lng):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&zoom=18&addressdetails=1"
        headers = {'User-Agent': 'MyCoolMotorcycleApp/1.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        address = data.get('address', {})
        return {
            "poi": address.get('amenity', address.get('shop', address.get('tourism', ''))),
            "road": address.get('road', ''),
            "suburb": address.get('suburb', address.get('city_district', ''))
        }
    except Exception as e:
        print(f"無法獲取地點名稱: {e}")
        return None


def find_closest_gps_point(event_timestamp_str, gps_trace):
    if not gps_trace or not event_timestamp_str: return None
    try:
        event_dt_utc = parser.isoparse(event_timestamp_str)
        taiwan_tz_offset = dt.timedelta(hours=8)
        event_dt_local = event_dt_utc + taiwan_tz_offset
        min_diff = float('inf')
        closest_point = None
        for point in gps_trace:
            gps_dt_pseudo_utc = parser.isoparse(point['timestamp'])
            diff = abs((event_dt_local - gps_dt_pseudo_utc).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_point = point
        if closest_point and min_diff < 15: return closest_point
    except Exception as e:
        print(f"Time match error: {e}")
    return None


@app.route('/get_groq_ai_response', methods=['POST'])
def get_groq_ai_api():
    if not GROQ_API_KEY or "gsk_" not in GROQ_API_KEY:
        return jsonify({"error": "請先在 app.py 設定正確的 GROQ_API_KEY"}), 500

    data = request.json
    danger_events = data.get('danger_events', [])
    gps_trace = data.get('gps_trace', [])

    location_data = data.get('location', {})  # 接收單點座標

    # --- 1. 統一計算分數 (Single Source of Truth) ---
    # 這是為了確保 AI 說的分數跟網頁圖表顯示的分數一模一樣
    danger_count = len(danger_events)
    calculated_score = 100 - (danger_count * 3)
    if calculated_score < 40:  # 設定保底分，避免分數太難看
        calculated_score = 40

    # 決定評語等級
    score_level = "優秀"
    if calculated_score < 60:
        score_level = "不及格 (需加油)"
    elif calculated_score < 80:
        score_level = "尚可 (可再改進)"

    # --- 2. 計算時段與時長 ---
    time_desc = "日間"
    duration_info = ""


    if gps_trace:
        try:
            start_time = parser.isoparse(gps_trace[0]['timestamp'])
            end_time = parser.isoparse(gps_trace[-1]['timestamp'])
            hour = (start_time.hour + 8) % 24
            if hour >= 18 or hour < 6: time_desc = "夜間"

            diff = int((end_time - start_time).total_seconds() / 60)
            duration_info = f"行程約 {diff} 分鐘"
        except:
            pass

    # --- 3. 設計 Prompt (強迫 AI 使用我們算好的分數) ---
    prompt = f"""
    角色設定：你是一位專業、親切且富有鼓勵性的「AI 駕駛教練」。
    情境：使用者剛完成一趟 {time_desc} 行駛，{duration_info}。

    【重要指令】：
    系統已經根據數據計算出本次駕駛分數為：{calculated_score} 分。
    請務必直接使用這個分數來撰寫報告，不要自己重新評分。

    請用繁體中文生成報告，適度使用 Emoji，包含以下區塊：

    1. 🛡️【駕駛安全評分】：
       - 請直接宣布：「本次駕駛評分：{calculated_score} 分」。
       - 評語：{score_level}。

    2. 👨‍🏫【教練講評】：
       - 針對 {calculated_score} 分給予對應的建議。
       - 如果是 40-59 分，請溫柔地鼓勵駕駛不要氣餒，提醒注意安全。
       - 結合「{time_desc}」的特性給予建議。

    3. 📍【危險路段分析】：
       - 針對發生最多次警報的地點，推測原因（如路口複雜 🚦、車流量大 🚗）。

    數據如下：
    """

    if danger_events and gps_trace:
        prompt += f"偵測到危險事件共 {danger_count} 次。\n"
        event_locations = defaultdict(int)
        for event in danger_events:
            closest_gps = find_closest_gps_point(event.get('timestamp'), gps_trace)
            if closest_gps:
                loc = get_location_name(closest_gps['lat'], closest_gps['lng'])
                if loc and (loc.get('road') or loc.get('suburb')):
                    event_locations[loc.get('road') or loc.get('suburb')] += 1

        if event_locations:
            prompt += "[危險熱點統計]:\n"
            for loc, count in event_locations.items():
                prompt += f"- {loc}: {count} 次\n"
        prompt += "\n"

    elif gps_trace:
        # 如果沒有危險事件，那就是 100 分
        prompt = prompt.replace(f"{calculated_score}", "100")
        start = gps_trace[0]
        end = gps_trace[-1]
        start_info = get_location_name(start['lat'], start['lng'])
        end_info = get_location_name(end['lat'], end['lng'])
        start_str = start_info.get('road', '起點') if start_info else "起點"
        end_str = end_info.get('road', '終點') if end_info else "終點"
        prompt += f"本次行程非常安全，沒有偵測到任何危險事件 👍。從 {start_str} 到 {end_str}。GPS 軌跡完整。\n"
    else:
        prompt += "數據不足（無 GPS 或危險事件）。請提醒使用者確認設備連接 🔌。\n"

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.5}

    try:
        # print("正在呼叫 Groq AI...")
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            error_msg = f"AI 服務回應錯誤: {response.status_code}"
            try:
                error_detail = response.json().get('error', {}).get('message', response.text)
                error_msg += f" - {error_detail}"
            except:
                pass
            return jsonify({"error": error_msg}), 500

        ai_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "AI 無回應")
        return jsonify({'aiResponse': ai_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 503


# --- 啟動 ---
if __name__ == '__main__':
    if not os.path.exists(video_database): save_videos_to_db([])
    start_buffer_cleanup_thread()
    # 確保 Host 是 0.0.0.0 讓外部可連
    app.run(debug=True, host="0.0.0.0", port=5000)