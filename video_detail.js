// video_detail.js (最終版 - 儀表板邏輯)

let detailMapInstance = null;
let gpsPolyline = null;
let gpsAccuracyChart = null;

window.initVideoDetailPageMap = function() {
    console.log("Google Maps API ready for detail page.");
    loadVideoDetails();
};

async function requestAiReport(gpsTrace, dangerEvents, displayElement) {
    if (!displayElement) return;
    displayElement.innerHTML = '<p class="loading-text">AI 正在分析行程數據...</p>';
    try {
        const response = await fetch('http://127.0.0.1:5000/get_groq_ai_response', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                gps_trace: gpsTrace || [],
                danger_events: dangerEvents || []
            })
        });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || `伺服器錯誤 ${response.status}`);
        }
        const data = await response.json();
        // 使用 <pre> 標籤以保留換行和空格，最穩定
        displayElement.innerHTML = marked.parse(data.aiResponse);
    } catch (error) {
        console.error("Failed to get AI report:", error);
        displayElement.innerHTML = `<p class="error-text">無法生成 AI 報告: ${error.message}</p>`;
    }
}

// 【新增】繪製 GPS 精準度圖表的函式
function renderGpsAccuracyChart(gpsTrace) {
    const chartContainer = document.getElementById('chartContainer');
    const chartPlaceholder = document.getElementById('chartPlaceholder');
    const ctx = document.getElementById('gpsAccuracyChart').getContext('2d');

    if (!gpsTrace || gpsTrace.length === 0) {
        chartPlaceholder.style.display = 'block';
        ctx.canvas.style.display = 'none';
        return;
    }

    chartPlaceholder.style.display = 'none';
    ctx.canvas.style.display = 'block';

    const labels = gpsTrace.map((_, index) => `點 ${index + 1}`);
    const accuracyData = gpsTrace.map(point => point.accuracy || 0);

    // 如果已存在圖表實例，先銷毀
    if (gpsAccuracyChart) {
        gpsAccuracyChart.destroy();
    }

    gpsAccuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'GPS 誤差值 (公尺)',
                data: accuracyData,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '誤差 (公尺)'
                    }
                },
                x: {
                     title: {
                        display: true,
                        text: '行程時間點'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `精準度: ${context.parsed.y.toFixed(2)} 公尺`;
                        }
                    }
                }
            }
        }
    });
}


// 【重構】主要載入函式
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
        const response = await fetch(`http://127.0.0.1:5000/get_video/${videoId}`);
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || `伺服器錯誤 ${response.status}`);
        }
        const videoData = await response.json();

        // --- 渲染內容 ---
        document.title = `行程詳情 - ${videoData.title}`;
        document.getElementById('videoDetailTitle').textContent = videoData.title;
        document.getElementById('videoDetailDate').textContent = `日期: ${videoData.date || "未知"}`;
        document.getElementById('videoDetailDescription').textContent = videoData.description || "無描述。";
        document.getElementById('videoDetailPlayerWrapper').innerHTML = videoData.content || '<p class="error-text">影片未能載入</p>';

        loadingMessage.style.display = 'none';
        videoContentDiv.style.display = 'block';

        // --- 非同步載入地圖、AI報告、圖表 ---
        if (videoData.gps_trace && videoData.gps_trace.length > 0) {
            displayGpsTraceMap(videoData.gps_trace, videoData.title);
        } else {
            document.getElementById('mapContainer').innerHTML = '<p class="placeholder-text">無 GPS 軌跡數據</p>';
        }

        requestAiReport(videoData.gps_trace, videoData.danger_events, document.getElementById('aiReportArea'));

        renderGpsAccuracyChart(videoData.gps_trace);

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
        detailMapInstance = new google.maps.Map(mapContainer, { gestureHandling: "cooperative" });
    }
    detailMapInstance.fitBounds(bounds);
    if (gpsPolyline) gpsPolyline.setMap(null);
    gpsPolyline = new google.maps.Polyline({ path, map: detailMapInstance, strokeColor: '#FF0000', strokeWeight: 4 });

    // 簡單起見，暫不重複添加標記，未來可優化
    if (path.length > 0) new google.maps.Marker({ position: path[0], map: detailMapInstance, label: "起" });
    if (path.length > 1) new google.maps.Marker({ position: path[path.length - 1], map: detailMapInstance, label: "終" });
}

// 頁面載入邏輯
document.addEventListener('DOMContentLoaded', () => {
    if (typeof google === 'undefined' || !google.maps) {
        console.log("DOM loaded, waiting for Maps API callback.");
    } else {
        console.log("DOM loaded and Maps API was ready, loading details.");
        loadVideoDetails();
    }
});