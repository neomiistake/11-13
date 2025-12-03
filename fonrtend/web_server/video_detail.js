// video_detail.js (最終版 - 含數據卡片與匯出功能)

const API_BASE_URL = ""; 

let detailMapInstance = null;
let gpsPolyline = null;
let gpsAccuracyChart = null;
let scoreChart = null;
let currentAiReportText = ""; // 暫存 AI 報告內容供匯出使用

window.initVideoDetailPageMap = function() {
    console.log("Google Maps API ready for detail page.");
    loadVideoDetails();
};

// --- 工具：計算地球上兩點距離 (Haversine formula) ---
function getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2) {
    var R = 6371; // 地球半徑 (km)
    var dLat = deg2rad(lat2 - lat1);
    var dLon = deg2rad(lon2 - lon1);
    var a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2);
    var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

function deg2rad(deg) {
    return deg * (Math.PI / 180)
}

// --- 工具：數字跳動動畫 ---
function animateValue(obj, start, end, duration, isFloat = false) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = progress * (end - start) + start;
        obj.innerHTML = isFloat ? value.toFixed(2) : Math.floor(value);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// --- 主要邏輯 ---

async function requestAiReport(gpsTrace, dangerEvents, displayElement) {
    if (!displayElement) return;
    displayElement.innerHTML = '<p class="loading-text" style="color:#007bff; font-weight:bold;">🤖 AI 正在分析行程數據...</p>';
    document.getElementById('downloadReportBtn').style.display = 'none';

    try {
        const response = await fetch(`${API_BASE_URL}/get_groq_ai_response`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                gps_trace: gpsTrace || [],
                danger_events: dangerEvents || []
            })
        });
        if (!response.ok) throw new Error(`伺服器錯誤 ${response.status}`);
        
        const data = await response.json();
        currentAiReportText = data.aiResponse; // 存下來給下載用
        displayElement.innerHTML = marked.parse(data.aiResponse);
        
        // 顯示下載按鈕
        if(currentAiReportText && currentAiReportText.length > 10) {
            document.getElementById('downloadReportBtn').style.display = 'block';
        }

    } catch (error) {
        console.error("Failed to get AI report:", error);
        displayElement.innerHTML = `<p class="error-text">無法生成 AI 報告: ${error.message}</p>`;
    }
}

// 下載報告功能
document.getElementById('downloadReportBtn').addEventListener('click', () => {
    if (!currentAiReportText) return;
    const blob = new Blob([currentAiReportText], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AI_Driving_Report_${new Date().getTime()}.txt`;
    a.click();
    window.URL.revokeObjectURL(url);
});

function calculateAndRenderStats(gpsTrace) {
    let totalDist = 0;
    let totalTimeMin = 0;
    let avgSpeed = 0;

    if (gpsTrace && gpsTrace.length > 1) {
        // 1. 計算距離
        for (let i = 0; i < gpsTrace.length - 1; i++) {
            totalDist += getDistanceFromLatLonInKm(
                gpsTrace[i].lat, gpsTrace[i].lng,
                gpsTrace[i+1].lat, gpsTrace[i+1].lng
            );
        }

        // 2. 計算時間 (假設 timestamp 是 ISO 字串)
        try {
            const start = new Date(gpsTrace[0].timestamp);
            const end = new Date(gpsTrace[gpsTrace.length - 1].timestamp);
            const diffMs = end - start;
            totalTimeMin = diffMs / (1000 * 60); // 分鐘
        } catch (e) { console.warn("Time calc error", e); }

        // 3. 計算平均速度 (km/h) = km / (min/60)
        if (totalTimeMin > 0) {
            avgSpeed = totalDist / (totalTimeMin / 60);
        }
    }

    // 執行數字動畫
    animateValue(document.getElementById('statDistance'), 0, totalDist, 1000, true);
    animateValue(document.getElementById('statDuration'), 0, totalTimeMin, 1000, true); // 改為顯示小數點後兩位
    animateValue(document.getElementById('statSpeed'), 0, avgSpeed, 1000, true);
}

function renderScoreChart(dangerCount) {
    const ctx = document.getElementById('scoreChart').getContext('2d');
    const scoreValueEl = document.getElementById('scoreValue');
    const scoreLabelEl = document.getElementById('scoreLabel');

    // 溫和版算分
    let score = 100 - (dangerCount * 3);
    if (score < 40) score = 40; 

    // 動畫跑分
    animateValue(scoreValueEl, 0, score, 1500);
    
    let scoreColor = '#28a745'; 
    let labelText = '駕駛表現優秀';
    
    if (score < 60) {
        scoreColor = '#dc3545'; 
        labelText = '需加強注意';
    } else if (score < 80) {
        scoreColor = '#ffc107'; 
        labelText = '可再改進';
    }
    
    scoreValueEl.style.color = scoreColor;
    scoreLabelEl.textContent = labelText;
    scoreLabelEl.style.color = scoreColor;

    if (scoreChart) scoreChart.destroy();

    scoreChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['得分', '扣分'],
            datasets: [{
                data: [score, 100 - score],
                backgroundColor: [scoreColor, '#e9ecef'],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            rotation: -90,
            circumference: 180,
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
        }
    });
}

function renderGpsAccuracyChart(gpsTrace) {
    const chartPlaceholder = document.getElementById('chartPlaceholder');
    const canvas = document.getElementById('gpsAccuracyChart');
    const ctx = canvas.getContext('2d');

    if (!gpsTrace || gpsTrace.length === 0) {
        chartPlaceholder.style.display = 'block';
        canvas.style.display = 'none';
        document.getElementById('scoreChart').style.display = 'none';
        document.getElementById('scoreValue').textContent = '-';
        return;
    }

    chartPlaceholder.style.display = 'none';
    canvas.style.display = 'block';

    const labels = gpsTrace.map((_, index) => index + 1);
    const accuracyData = gpsTrace.map(point => point.accuracy || 0);

    if (gpsAccuracyChart) gpsAccuracyChart.destroy();

    gpsAccuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'GPS 誤差 (m)',
                data: accuracyData,
                borderColor: '#17a2b8',
                backgroundColor: 'rgba(23, 162, 184, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, grid: { color: '#f0f0f0' }, ticks: { font: { size: 10 } } },
                x: { display: false }
            },
            plugins: { legend: { display: true, labels: { boxWidth: 10, font: { size: 11 } } } }
        }
    });
}

async function loadVideoDetails() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessageElement = document.getElementById('errorMessage');
    const videoContentDiv = document.getElementById('videoContent');
    const urlParams = new URLSearchParams(window.location.search);
    const videoId = urlParams.get('id');

    if (!videoId) {
        loadingMessage.style.display = 'none';
        errorMessageElement.textContent = '錯誤：URL中未提供有效的影片 ID。';
        errorMessageElement.style.display = 'block';
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/get_video/${videoId}`);
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || `伺服器錯誤 ${response.status}`);
        }
        const videoData = await response.json();

        document.title = `行程詳情 - ${videoData.title}`;
        document.getElementById('videoDetailTitle').textContent = videoData.title;
        document.getElementById('videoDetailDate').textContent = `日期: ${videoData.date || "未知"}`;
        document.getElementById('videoDetailDescription').textContent = videoData.description || "無描述。";
        
        const playerWrapper = document.getElementById('videoDetailPlayerWrapper');
        if (videoData.content) {
            playerWrapper.innerHTML = videoData.content;
            const videoEl = playerWrapper.querySelector('video');
            if (videoEl) {
                videoEl.style.width = '100%';
                videoEl.style.height = '100%';
                videoEl.style.maxHeight = '400px';
            }
        } else {
            playerWrapper.innerHTML = '<p class="error-text" style="color:white;">影片未能載入</p>';
        }

        loadingMessage.style.display = 'none';
        videoContentDiv.style.display = 'block';

         // 情況 A：有完整的 GPS 軌跡 (自動錄影產生的) -> 畫路線
        if (videoData.gps_trace && videoData.gps_trace.length > 0) {
            displayGpsTraceMap(videoData.gps_trace, videoData.title);
        }
        // 情況 B：沒有軌跡，但有手動選擇的單一地點 -> 畫單點
        else if (videoData.location && videoData.location.lat && videoData.location.lng) {
            displaySingleLocationMap(videoData.location, videoData.title);
        }
        // 情況 C：什麼都沒有
        else {
            document.getElementById('mapContainer').innerHTML = '<p class="placeholder-text">無 GPS 數據或地點資訊</p>';
        }

        // 呼叫新功能
        calculateAndRenderStats(videoData.gps_trace);
        requestAiReport(videoData.gps_trace, videoData.danger_events, document.getElementById('aiReportArea'));
        renderGpsAccuracyChart(videoData.gps_trace);
        renderScoreChart((videoData.danger_events || []).length);

    } catch (error) {
        console.error("Error loading video details:", error);
        loadingMessage.style.display = 'none';
        errorMessageElement.textContent = `載入影片詳情時發生錯誤: ${error.message}`;
        errorMessageElement.style.display = 'block';
    }
}

function displayGpsTraceMap(gpsTrace, videoTitle = '影片路線') {
    const mapContainer = document.getElementById('mapContainer');
    if (!google || !google.maps) {
        mapContainer.innerHTML = '<p class="map-error">地圖功能不可用</p>';
        return;
    }
    const path = gpsTrace.map(p => ({ lat: p.lat, lng: p.lng }));
    const bounds = new google.maps.LatLngBounds();
    path.forEach(p => bounds.extend(p));
    
    if (!detailMapInstance) {
        detailMapInstance = new google.maps.Map(mapContainer, { 
            gestureHandling: "cooperative",
            mapTypeId: google.maps.MapTypeId.ROADMAP
        });
    }
    
    detailMapInstance.fitBounds(bounds);
    
    if (gpsPolyline) gpsPolyline.setMap(null);
    
    gpsPolyline = new google.maps.Polyline({ 
        path: path, 
        map: detailMapInstance, 
        strokeColor: '#007bff', 
        strokeOpacity: 1.0, 
        strokeWeight: 4 
    });

    if (path.length > 0) {
        new google.maps.Marker({ position: path[0], map: detailMapInstance, label: { text: "起", color: "white" } });
    }
    if (path.length > 1) {
        new google.maps.Marker({ position: path[path.length - 1], map: detailMapInstance, label: { text: "終", color: "white" } });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (typeof google === 'undefined' || !google.maps) {
        console.log("DOM loaded, waiting for Maps API callback.");
    } else {
        console.log("DOM loaded and Maps API was ready, loading details.");
        loadVideoDetails();
    }
});



// --- 新增：顯示單一地點的地圖 (給手動新增影片用) ---
function displaySingleLocationMap(location, title) {
    const mapContainer = document.getElementById('mapContainer');

    // 檢查 Google Maps API
    if (!google || !google.maps) {
        mapContainer.innerHTML = '<p class="map-error">地圖載入失敗</p>';
        return;
    }

    // 解析經緯度 (確保是數字)
    const lat = parseFloat(location.lat);
    const lng = parseFloat(location.lng);

    // 建立地圖
    const map = new google.maps.Map(mapContainer, {
        center: { lat: lat, lng: lng },
        zoom: 15, // 單點顯示時，Zoom 近一點比較清楚
        mapTypeId: google.maps.MapTypeId.ROADMAP
    });

    // 插上一根紅色的針
    new google.maps.Marker({
        position: { lat: lat, lng: lng },
        map: map,
        title: title || "影片地點",
        animation: google.maps.Animation.DROP // 掉落動畫
    });
}