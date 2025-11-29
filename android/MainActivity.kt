package com.example.webweb

import android.Manifest
import android.content.Intent //使用 Intent 打開 Google Maps 應用程式
import android.content.pm.PackageManager
import android.location.Location
import android.net.Uri
import android.os.Bundle
import android.os.Looper
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat


import com.google.android.gms.location.FusedLocationProviderClient  //使用 Google Play Services 的 FusedLocationProviderClient 取得位置資料
import com.google.android.gms.location.LocationCallback //使用 LocationCallback 接收位置更新
import com.google.android.gms.location.LocationRequest //使用 LocationRequest 設定位置更新的參數
import com.google.android.gms.location.LocationResult//使用 LocationResult 獲取位置更新結果
import com.google.android.gms.location.LocationServices//使用 LocationServices 獲取 FusedLocationProviderClient 實例
import com.google.android.material.switchmaterial.SwitchMaterial
import com.google.firebase.ktx.Firebase //使用 Firebase 雲端訊息服務 (FCM) 訂閱和取消訂閱主題
import com.google.firebase.messaging.ktx.messaging //使用 Firebase 雲端訊息服務 (FCM) 訂閱和取消訂閱主題
import com.google.gson.Gson //使用 Gson 將 Kotlin Map 轉換為 JSON 字串，以便發送給後端 API
import com.google.firebase.messaging.FirebaseMessaging


import kotlinx.coroutines.CoroutineScope //使用 CoroutineScope 進行非同步網路請求 可以避免阻塞主線程
import kotlinx.coroutines.Dispatchers //使用 Dispatchers.IO 在背景執行網路請求 可以避免阻塞主線程
import kotlinx.coroutines.Job //使用 Job 管理協程任務 可以在需要時取消任務
import kotlinx.coroutines.delay //使用 delay 在協程中進行非阻塞延遲
import kotlinx.coroutines.launch //使用 launch 啟動協程  可以在背景執行網路請求


import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient     //使用 OkHttpClient 連線 Flask 後端 API (/receive_gps_data, /get_current_recording_session_id)  兩個端點
import okhttp3.Request //使用 Request 建立 HTTP 請求
//這裡詳細解釋一下 HTTP就是網路通訊協定  他有很多種方法  例如 GET POST PUT DELETE 等等
//GET 是從伺服器取得資料  POST 是向伺服器提交資料
import okhttp3.RequestBody.Companion.toRequestBody //使用 toRequestBody 將 JSON 字串轉換為請求體 以便發送給後端 API


import java.io.IOException //處理網路請求中的 IO 異常
import java.text.SimpleDateFormat //使用 SimpleDateFormat 將時間戳轉換為 ISO 8601 格式的字串
import java.util.Date //使用 Date 獲取位置的時間戳
import java.util.Locale //使用 Locale.US 指定時間格式的區域設定 這個可以確保時間格式的一致性




class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"
    private val LOCATION_PERMISSION_REQUEST_CODE = 1001  //位置權限請求代碼 1001代表請求位置權限

    private lateinit var switchNotifications: SwitchMaterial //這是前端的開關元件 用來開啟或關閉通知
    private lateinit var btnOpenMaps: Button //這是打開 google map 的按鈕 注意 這是外部的 不是內嵌的
    private lateinit var btnStartStopGPS: Button //這是開始或停止 gps 追蹤的按鈕

    private val DANGER_TOPIC = "danger_alerts"  //這名稱請去看 firebase console 裡面的雲端訊息服務(Firebase Cloud Messaging) 裡面的主題名稱

    private lateinit var fusedLocationClient: FusedLocationProviderClient //用來取得位置資料的客戶端  意思是 取得手機的 gps 定位
    private lateinit var locationCallback: LocationCallback //用來接收位置更新的回調函式 意思是 當位置有更新時會呼叫這個函式
    private var isTrackingLocation = false //是否正在追蹤位置的標誌

    private val client = OkHttpClient() //建立一個 OkHttpClient 實例 用來連線後端 api
    private val JSON = "application/json; charset=utf-8".toMediaType() //設定請求體的媒體類型為 JSON 為什麼要寫這個 因為我們要發送的資料是 json 格式的

    // **重要：請根據你的 ZeroTier 或內網 IP 修改這裡**
    private val FLASK_BACKEND_BASE_URL = "http://192.168.196.207:5000"  //這是後端那邊的ip位置 包含端口5000 兩個端點在閜面那兩個
    private val RECEIVE_GPS_ENDPOINT = "$FLASK_BACKEND_BASE_URL/receive_gps_data" //POST 這邊提交gps數據
    private val GET_SESSION_ID_ENDPOINT = "$FLASK_BACKEND_BASE_URL/get_current_recording_session_id" //GET 這邊拿後端那邊的id(trip id)

    private var currentTrackingSessionId: String? = null // 當前用於 GPS 追蹤的 Session ID

    private var sessionCheckJob: Job? = null //確認id是否還在工作
    private val ACCURACY_THRESHOLD_METERS = 20.0; //設定一個準確度閾值  單位是公尺
    override fun onCreate(savedInstanceState: Bundle?) {//活動建立時呼叫
        super.onCreate(savedInstanceState)//呼叫父類別的 onCreate 方法
        setContentView(R.layout.activity_main)//設定活動的佈局檔案
        FirebaseMessaging.getInstance().token.addOnCompleteListener { task ->
            if (!task.isSuccessful) {
                Log.w(TAG, "Fetching FCM registration token failed", task.exception)
                return@addOnCompleteListener
            }
            val token = task.result
            // 打印到 Logcat，方便複製
            Log.d(TAG, "!!!!!!!!!!!!!!!!!! COPY THIS TOKEN !!!!!!!!!!!!!!!!!!")
            Log.d(TAG, token)
            Log.d(TAG, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        }

        switchNotifications = findViewById(R.id.switchNotifications) //初始化 UI 元件
        btnOpenMaps = findViewById(R.id.btnOpenMaps) //初始化 UI 元件
        btnStartStopGPS = findViewById(R.id.btnStartStopGPS) //初始化 UI 元件

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this) //初始化 FusedLocationProviderClient 實例 用來取得位置資料

        locationCallback = object : LocationCallback() { //初始化 LocationCallback 實例 用來接收位置更新
            override fun onLocationResult(locationResult: LocationResult) { //當有位置更新時呼叫
                locationResult.lastLocation?.let { location ->//取得最新的位置
                    if (location.accuracy > ACCURACY_THRESHOLD_METERS) {  //他是誤差值而不是準確率  誤差值>xxx 而不動作這樣
                        // 如果準確度太差，就打印一條警告日誌，然後直接 return，中止後續操作。
                        Log.w(TAG, "GPS accuracy poor (${location.accuracy}m > ${ACCURACY_THRESHOLD_METERS}m), point discarded.")
                        return@let // 使用 @let 來確保我們只跳出這個 let 區塊
                    }
                    Log.d(TAG, "Location received: Lat=${location.latitude}, Lng=${location.longitude}")
                    // 只有當我們有 Session ID 時才發送
                    currentTrackingSessionId?.let { sessionId ->
                        sendLocationDataToBackend(location, sessionId) //發送位置資料到後端 包含 session id 經緯度 時間戳 準確度
                    } ?: run {
                        Log.w(TAG, "No active session ID to send GPS data with.")
                    }
                }
            }
        }

        //設定開關元件的監聽器 這開關是用來開啟或關閉通知的
        switchNotifications.setOnCheckedChangeListener { _, isChecked -> //當開關狀態改變時呼叫
            if (isChecked) {
                Firebase.messaging.subscribeToTopic(DANGER_TOPIC).addOnCompleteListener { task ->  //訂閱主題 使用 Firebase Cloud Messaging
                    val msg = if (task.isSuccessful) "已開啟警報通知" else "開啟通知失敗"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                }
            } else {
                Firebase.messaging.unsubscribeFromTopic(DANGER_TOPIC).addOnCompleteListener { task ->
                    val msg = if (task.isSuccessful) "已關閉警報通知" else "關閉通知失敗"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                }
            }
        }


        //設定按鈕的監聽器 打開 google map 的按鈕
        btnOpenMaps.setOnClickListener { //打開 google map  記住這是外部的 不是內嵌的
            val gmmIntentUri = Uri.parse("geo:0,0?q=") //建立一個 geo URI 用來打開地圖 這個 URI 會打開地圖並顯示當前位置 URI是 geo:0,0?q= 他可以接受經緯度參數 也可以接受地址參數
            val mapIntent = Intent(Intent.ACTION_VIEW, gmmIntentUri) //建立一個 Intent 用來打開地圖應用程式
            mapIntent.setPackage("com.google.android.apps.maps") //指定要打開的應用程式是 Google Maps
            if (mapIntent.resolveActivity(packageManager) != null) { //檢查是否有應用程式可以處理這個 Intent
                startActivity(mapIntent) //啟動地圖應用程式
            } else {
                Toast.makeText(this, "請先安裝 Google Maps", Toast.LENGTH_SHORT).show()
            }
        }

        //設定按鈕的監聽器 開始或停止 gps 追蹤的按鈕
        btnStartStopGPS.setOnClickListener {
            if (isTrackingLocation) {
                stopLocationUpdates() //如果正在追蹤位置就停止位置更新
            } else {
                getAndStartTrackingSession() //如果沒有在追蹤位置就取得並開始追蹤 session id
            }
        }
    }
    //-----------------------------------------------------------------------------------------------------
    //解釋一下 IO就是 input output  他是用來處理檔案讀寫 網路請求等操作的 網路請求、檔案讀寫、資料庫操作等
    //要切到主線程的原因是 因為我們在協程裡面  協程是背景線程 不能直接更新 UI 所以要用 runOnUiThread
    //會有背景協程的原因是 因為網路請求是耗時操作 不能阻塞主線程
    //但要切回主線程更新 UI 例如 Toast 或按鈕文字 就要用 runOnUiThread 要切回主線程!!!!!
    //-----------------------------------------------------------------------------------------------------

    private fun getAndStartTrackingSession() { //取得並開始追蹤 session id
        Toast.makeText(this, "正在查詢錄影 trip ID...", Toast.LENGTH_SHORT).show()
        CoroutineScope(Dispatchers.IO).launch { //在背景執行網路請求 為什麼要背景 因為網路請求是耗時操作 不能阻塞主線程
            try {
                val request = Request.Builder().url(GET_SESSION_ID_ENDPOINT).get().build() //建立一個 GET 請求 會去後端拿 session id
                val response = client.newCall(request).execute() //執行請求並獲取響應

                if (response.isSuccessful) {
                    val jsonResponse = response.body?.string() //取得響應的 JSON 字串
                    val gson = Gson()  //將 Kotlin Map 與 JSON 字串互轉
                    val result = gson.fromJson(jsonResponse, Map::class.java) //能夠把 Java/Kotlin 物件和 JSON 格式之間進行互相轉換
                    val sessionId = result["session_id"] as? String //從響應中提取 session_id

                    //所以上面這些都是在做網路請求 拿 session id 的動作
                    //這些動作都是在背景線程執行的


                    if (sessionId != null && sessionId.isNotEmpty()) { //如果 session_id 不為空就開始追蹤
                        currentTrackingSessionId = sessionId //儲存當前用於 GPS 追蹤的 Session ID
                        runOnUiThread {
                            //切換到主線程更新 UI 這裡比較複雜 因為我們在協程裡面  協程是背景線程 不能直接更新 UI 所以要用 runOnUiThread
                            //協成意思是 你可以把它想像成一個輕量級的線程 它可以讓你在背景執行一些耗時的操作 而不會阻塞主線程
                            //線程是系統分配給應用程式的執行單位  它有自己的堆疊和執行狀態
                            startLocationUpdates()//開始位置更新
                            Toast.makeText(this@MainActivity, "已獲取 Session ID: $sessionId", Toast.LENGTH_SHORT).show()
                        }
                    } else {
                        runOnUiThread {//thread 主線程 為什麼要這樣寫 因為我們在協程裡面  協程是背景線程 不能直接更新 UI 所以要用 runOnUiThread   那為什麼要背景線程的原因是 因為網路請求是耗時操作 不能阻塞主線程
                            Toast.makeText(this@MainActivity, "目前沒有活躍的錄影 Session ID，請先在電腦端啟動錄影程式。", Toast.LENGTH_LONG).show()  //Toast show他是屬於UI更新
                            Log.w(TAG, "No active recording session ID found.")
                        }
                    }
                } else { //如果響應不成功就顯示錯誤訊息
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "查詢 Session ID 失敗: ${response.code}", Toast.LENGTH_LONG).show()
                        Log.e(TAG, "Failed to get session ID: ${response.body?.string()}")
                    }
                }
            } catch (e: IOException) { //catch是捕捉異常的意思
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "網路錯誤: 無法查詢 Session ID(檢查zerotier)", Toast.LENGTH_LONG).show()
                    Log.e(TAG, "Network error getting session ID: ${e.message}")
                }
            } catch (e: Exception) {//捕捉其他異常
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "發生錯誤: ${e.message}", Toast.LENGTH_LONG).show()
                    Log.e(TAG, "Unexpected error getting session ID: ${e.message}")
                }
            }
        }
    }

    private fun startLocationUpdates() { //開始位置更新
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION), LOCATION_PERMISSION_REQUEST_CODE)
            //要求位置權限 requestPermissions checkselfPermission 是用來檢查是否有權限的 requestPermissions 是用來請求權限的
            return
        }

        val locationRequest = LocationRequest.Builder(5000L) // 每 5 秒更新一次
            .setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY)
            .setMinUpdateIntervalMillis(2000L) // 最快每 2 秒更新一次
            .build()

        //開始請求位置更新
        fusedLocationClient.requestLocationUpdates(locationRequest, locationCallback, Looper.getMainLooper()) //Looper.getMainLooper() 確保回調在主線程執行
        isTrackingLocation = true //設定正在追蹤位置的標誌為 true
        btnStartStopGPS.text = "停止 GPS 追蹤 (ID: ${currentTrackingSessionId?.substring(0,8)}...)"
        Toast.makeText(this, "開始 GPS 追蹤...", Toast.LENGTH_SHORT).show()
        Log.d(TAG, "Started GPS tracking for session: $currentTrackingSessionId")

        //啟動一個背景協程，每 10 秒檢查一次 Session 是否還活躍
        sessionCheckJob?.cancel() // 先取消可能存在的舊任務
        sessionCheckJob = CoroutineScope(Dispatchers.IO).launch {
            while (isTrackingLocation) {
                delay(10000) // 等待 10 秒
                checkSessionStatus() //檢查 Session 狀態
            }
        }
    }

    private fun stopLocationUpdates() {//停止位置更新
        sessionCheckJob?.cancel() //當手動停止或自動停止時，確保取消背景檢查任務

        fusedLocationClient.removeLocationUpdates(locationCallback) //移除位置更新請求
        isTrackingLocation = false //設定正在追蹤位置的標誌為 false
        btnStartStopGPS.text = "開始 GPS 追蹤"
        Toast.makeText(this, "停止 GPS 追蹤", Toast.LENGTH_SHORT).show()
        Log.d(TAG, "Stopped GPS tracking for session: $currentTrackingSessionId")
        currentTrackingSessionId = null // 停止後清除 Session ID
    }
    /////////////////////////////////////////////測試 id 檢查
    private fun checkSessionStatus() {//檢查 Session 狀態
        val sessionIdToCheck = currentTrackingSessionId ?: return // 如果當前沒有在追蹤的ID，就直接返回

        CoroutineScope(Dispatchers.IO).launch { //在背景執行網路請求
            try {
                val request = Request.Builder().url(GET_SESSION_ID_ENDPOINT).get().build() //建立一個 GET 請求 會去後端拿 session id
                val response = client.newCall(request).execute() //執行請求並獲取響應
                if (response.isSuccessful) { //如果響應成功
                    val jsonResponse = response.body?.string() //取得響應的 JSON 字串
                    val result = Gson().fromJson(jsonResponse, Map::class.java) //從json變形    Kotlin Map 與 JSON 字串互轉
                    //這裡從flask後端拿到的 session id 這id是json格式的 所以用 Gson 轉成 Map
                    //所以result 這個變數就是一個 Map 物件
                    val activeSessionIdFromServer = result["session_id"] as? String //從響應中提取 session_id

                    // 核心邏輯：如果伺服器上最新的活躍ID 和 我們手機上正在追蹤的ID 不一樣了
                    // (通常是因為伺服器上的變成了 null)
                    if (activeSessionIdFromServer != sessionIdToCheck) {
                        runOnUiThread {
                            Log.d(TAG, "Session '$sessionIdToCheck' is no longer active on server. Stopping tracking automatically.")
                            Toast.makeText(this@MainActivity, "錄影已結束，自動停止GPS追蹤", Toast.LENGTH_LONG).show()
                            stopLocationUpdates() // 呼叫停止函式，實現自動斷開！
                        }
                    }
                }
            } catch (e: Exception) {//捕捉其他異常
                Log.e(TAG, "Error checking session status: ${e.message}")
            }
        }
    }
    /////////////////////////////////////////////測試 id 檢查
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) { //處理權限請求結果
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)//呼叫父類別的 onRequestPermissionsResult 方法
        //onRequestPermissionsResult 是當使用者回應權限請求時會被呼叫
        //這裡呼叫他是因為我們要處理使用者的回應
        if (requestCode == LOCATION_PERMISSION_REQUEST_CODE) { //如果請求代碼是位置權限請求代碼
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                getAndStartTrackingSession()//如果權限被授予就取得並開始追蹤 session id
            } else {
                Toast.makeText(this, "位置權限被拒絕，無法追蹤 GPS。", Toast.LENGTH_LONG).show() //如果權限被拒絕就顯示錯誤訊息
                Log.w(TAG, "Location permission denied.") //記錄警告日誌
            }
        }
    }

    private fun sendLocationDataToBackend(location: Location, sessionId: String) { //發送位置數據到後端 資料包含 session id 經緯度 時間戳 準確度
        val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US) // ISO 8601 年月日時間格式
        val timestamp = dateFormat.format(Date(location.time)) //將位置時間轉換為 ISO 8601 格式的字串

        val gpsData = mapOf( //建立一個包含 GPS 資料的 Map
            "session_id" to sessionId, //這個是用來區分不同錄影的id
            "latitude" to location.latitude, //經度
            "longitude" to location.longitude, //緯度
            "timestamp" to timestamp, //時間戳
            "accuracy" to location.accuracy //準確度
        )
        val json = Gson().toJson(gpsData) //將 Map 轉換為 JSON 字串 把gpsData 變成 json 字串 這樣後端flasks那邊才看得懂
        // Log.d(TAG, "Sending GPS data: $json")

        CoroutineScope(Dispatchers.IO).launch { //CoroutineScope 讓網路請求在背景執行 Dispatchers.IO 適合進行阻塞型 I/O 操作，例如網路請求或檔案讀寫
            try {
                val request = Request.Builder() //建立一個 POST 請求
                    .url(RECEIVE_GPS_ENDPOINT) //目標端點 /receive_gps_data
                    .post(json.toRequestBody(JSON)) //將 JSON 字串作為請求體發送
                    .build()//建立請求物件

                val response = client.newCall(request).execute() //這裡把請求發送出去並獲取響應 響應就是後端回傳給你的東西
                if (!response.isSuccessful) {
                    Log.e(TAG, "Failed to send GPS data. Code: ${response.code}, Response: ${response.body?.string()}")
                }
            } catch (e: IOException) {
                Log.e(TAG, "Network error sending GPS data: ${e.message}")
            } catch (e: Exception) {
                Log.e(TAG, "Unexpected error sending GPS data: ${e.message}")
            }
        }
    }

    override fun onDestroy() { //當活動被銷毀時停止位置更新 就是錄影程式結束時
        super.onDestroy()
        stopLocationUpdates()
    }
}