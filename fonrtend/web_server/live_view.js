// live_view.js (最終版 - 危險曲線 + 自動快照)

document.addEventListener('DOMContentLoaded', () => {
    // --- 配置 ---
    const mediamtxBaseUrl = "http://192.168.196.73:8889";
    const streamPaths = ["cam0", "cam1"];
    const API_BASE_URL = ""; 

    // --- DOM 元素 ---
    const systemStatusBadge = document.getElementById('systemStatusBadge');
    const currentTripIdEl = document.getElementById('currentTripId');
    const alertLogArea = document.getElementById('alertLogArea');
    const videoCountEl = document.getElementById('videoCount');
    const alertCountEl = document.getElementById('alertCount');
    const liveTimeEl = document.getElementById('liveTime');
    const liveDateEl = document.getElementById('liveDate');
    
    // 新增元素
    const lastSnapshotArea = document.getElementById('lastSnapshotArea');
    const lastSnapshotTime = document.getElementById('lastSnapshotTime');

    let knownAlertIds = new Set();
    let dangerChart = null;
    
    // 用來記錄「當下」的危險值，0=無，1=有
    let currentDangerLevel = 0; 

    // --- 1. 時鐘功能 ---
    function updateClock() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('zh-TW', { hour12: false, hour:'2-digit', minute:'2-digit', second:'2-digit' });
        const dateStr = now.toISOString().split('T')[0];
        if(liveTimeEl) liveTimeEl.textContent = timeStr;
        if(liveDateEl) liveDateEl.textContent = dateStr;
    }
    setInterval(updateClock, 1000);
    updateClock();

    // --- 2. 初始化危險趨勢圖表 ---
    function initDangerChart() {
        const canvas = document.getElementById('dangerChart');
        if(!canvas) return;
        
        const ctx = canvas.getContext('2d');
        // 初始化 60 個數據點 (代表過去 60 秒)
        const initialData = Array(60).fill(0);
        const labels = Array(60).fill('');
        
        dangerChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: '危險指數',
                    data: initialData,
                    borderColor: '#dc3545', // 紅色
                    backgroundColor: 'rgba(220, 53, 69, 0.2)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4, // 平滑曲線
                    pointRadius: 0 // 不顯示點
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 0 }, // 關閉動畫以求即時反應
                scales: {
                    y: { 
                        min: 0, 
                        max: 1.2, 
                        ticks: { stepSize: 1, callback: (val) => val === 1 ? '危險' : '安全' },
                        grid: { color: '#f0f0f0' }
                    },
                    x: { display: false }
                },
                plugins: { legend: { display: false } }
            }
        });

        // 每秒更新圖表
        setInterval(() => {
            if(!dangerChart) return;
            
            dangerChart.data.datasets[0].data.push(currentDangerLevel);
            dangerChart.data.datasets[0].data.shift(); // 移除最舊的
            dangerChart.update();
            
            // 讓危險值慢慢冷卻歸零 (製造脈衝效果)
            if (currentDangerLevel > 0) {
                // 下一秒自動歸零，形成一個尖峰
                currentDangerLevel = 0; 
            }
        }, 1000);
    }
    initDangerChart();

    // --- 3. 自動截圖功能 (當警報發生時呼叫) ---
    function captureEvidence() {
        // 優先截取 cam0，若失敗則試 cam1
        const video = document.getElementById('video-cam0') || document.getElementById('video-cam1');
        if (!video || video.paused || video.ended) return;

        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth; 
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // 加上紅色邊框與文字，增加警示感
        ctx.strokeStyle = "red";
        ctx.lineWidth = 8;
        ctx.strokeRect(0, 0, canvas.width, canvas.height);
        
        ctx.font = "bold 24px Arial";
        ctx.fillStyle = "red";
        ctx.fillText("⚠️ DANGER DETECTED", 20, 40);
        
        // 加上時間戳
        ctx.font = "16px Consolas";
        ctx.fillStyle = "white";
        ctx.fillText(new Date().toLocaleString(), 20, canvas.height - 20);

        const dataURL = canvas.toDataURL("image/png");
        
        // 更新到畫面上
        if(lastSnapshotArea) {
            lastSnapshotArea.innerHTML = `<img src="${dataURL}" class="snapshot-img" alt="Evidence">`;
        }
        if(lastSnapshotTime) {
            lastSnapshotTime.textContent = "截圖時間: " + new Date().toLocaleTimeString('zh-TW');
        }
    }

    // --- 4. WebRTC 連接 ---
    async function connectStream(path) {
        const videoElement = document.getElementById(`video-${path}`);
        const statusElement = document.getElementById(`status-${path}`);
        const dotEl = document.getElementById(`dot-${path}`);
        const textEl = document.getElementById(`text-${path}`);

        if (!videoElement || !statusElement) return;
        const pc = new RTCPeerConnection();

        pc.oniceconnectionstatechange = () => {
            const state = pc.iceConnectionState;
            statusElement.textContent = `狀態: ${state}`;
            if (state === 'connected' || state === 'completed') {
                statusElement.style.opacity = 0;
                if(dotEl) dotEl.className = 'dot dot-green';
                if(textEl) textEl.textContent = '訊號良好';
            } else if (state === 'disconnected') {
                statusElement.style.opacity = 1;
                if(dotEl) dotEl.className = 'dot dot-red';
                if(textEl) textEl.textContent = '訊號中斷';
            }
        };

        pc.ontrack = (event) => {
            if (event.track.kind === 'video') videoElement.srcObject = event.streams[0];
        };

        try {
            pc.addTransceiver('video', { 'direction': 'recvonly' });
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            const response = await fetch(`${mediamtxBaseUrl}/${path}/whep`, {
                method: 'POST', headers: { 'Content-Type': 'application/sdp' }, body: offer.sdp
            });
            if (response.status !== 201) throw new Error(`Status ${response.status}`);
            const answerSdp = await response.text();
            await pc.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp: answerSdp }));
        } catch (error) {
            statusElement.textContent = `等待訊號...`;
            if(dotEl) dotEl.className = 'dot dot-red';
            if(textEl) textEl.textContent = '無訊號';
            setTimeout(() => connectStream(path), 5000);
        }
    }

    // --- 5. Dashboard 數據輪詢 (關鍵修改) ---
    async function fetchDashboardStats() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/dashboard/stats`);
            if (!response.ok) return;
            const data = await response.json();
            updateDashboardUI(data);
        } catch (error) { console.error("Poll error", error); }
    }

    function updateDashboardUI(data) {
        // 更新 AI 狀態標籤
        const isRecording = (data.status === 'recording');
        streamPaths.forEach(path => {
            const badge = document.getElementById(`badges-${path}`);
            if (badge) isRecording ? badge.classList.add('ai-active') : badge.classList.remove('ai-active');
        });

        if (isRecording) {
            systemStatusBadge.className = 'status-badge status-recording';
            systemStatusBadge.innerHTML = '● 錄影監控中';
        } else {
            systemStatusBadge.className = 'status-badge status-online';
            systemStatusBadge.textContent = '待機中';
        }

        currentTripIdEl.textContent = data.session_id || "等待連線...";
        videoCountEl.textContent = data.total_videos || "-";
        
        const alerts = data.alerts || [];
        alertCountEl.textContent = alerts.length;

        if (alerts.length > 0) {
            // 檢查是否有「新」警報
            const currentIds = new Set(alerts.map(a => a.timestamp + a.description));
            // 只要數量變多，或是有新的 ID 出現，就視為新警報
            if (currentIds.size > knownAlertIds.size || [...currentIds].some(id => !knownAlertIds.has(id))) {
                
                // 🔥 觸發危險反應：拉高圖表、截圖
                triggerDangerAlert();
                
                renderAlerts(alerts);
                knownAlertIds = currentIds;
            }
        } else {
            if (!alertLogArea.innerHTML.includes("尚無危險警報")) {
                alertLogArea.innerHTML = '<div style="text-align: center; color: #999; padding: 20px;">尚無危險警報</div>';
                knownAlertIds.clear();
            }
        }
    }

    function triggerDangerAlert() {
        console.log("🔥 危險警報觸發！");
        // 1. 拉高圖表指數到 1
        currentDangerLevel = 1; 
        // 2. 執行自動截圖
        captureEvidence();
    }

    function renderAlerts(alerts) {
        alertLogArea.innerHTML = '';
        alerts.forEach(alert => {
            const div = document.createElement('div');
            div.className = 'alert-item';
            div.innerHTML = `<span class="alert-time">${new Date(alert.timestamp).toLocaleTimeString('zh-TW', { hour12: false })}</span><span class="alert-msg">${alert.description}</span>`;
            alertLogArea.appendChild(div);
        });
    }

    // --- 啟動 ---
    streamPaths.forEach(path => connectStream(path));
    setInterval(fetchDashboardStats, 1000);
    fetchDashboardStats();
    
    // 手動功能保留
    window.takeSnapshot = function(path) {
        const video = document.getElementById(`video-${path}`);
        if (!video) return;
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0);
        const a = document.createElement("a");
        a.href = canvas.toDataURL("image/png");
        a.download = "snapshot.png"; a.click();
    };
    window.toggleFullscreen = function(id) {
        const e = document.getElementById(id);
        if(document.fullscreenElement) document.exitFullscreen();
        else e.requestFullscreen();
    };
});