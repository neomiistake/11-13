// script.js (最終美化版 - 戰情室儀表板)

window.initAppMaps = function() {
    console.log("Google Maps API ready for index.html.");
};

document.addEventListener('DOMContentLoaded', async () => {
    // --- 配置 ---
    const API_BASE_URL = ""; 

    // --- DOM 元素 ---
    const searchInput = document.getElementById('searchInput');
    const videoPlayerArea = document.getElementById('videoPlayerArea');
    const aiResponseTextArea = document.getElementById('aiResponseTextArea');
    
    // 新增：數據總覽元素
    const totalAvgScoreEl = document.getElementById('totalAvgScore');
    const totalTripsEl = document.getElementById('totalTrips');
    const totalDangersEl = document.getElementById('totalDangers');

    // 表單相關
    const showAddVideoButton = document.getElementById('showAddVideoButton');
    const addVideoFormContainer = document.getElementById('addVideoFormContainer');
    const addVideoForm = document.getElementById('addVideoForm');
    const cancelAddVideoButton = document.getElementById('cancelAddVideoButton');
    
    // 地圖選點相關
    const addVideoMapDiv = document.getElementById('addVideoMap');
    const locationSearchInput = document.getElementById('locationSearchInput');
    const selectedLatInput = document.getElementById('selectedLat');
    const selectedLngInput = document.getElementById('selectedLng');
    const displayLatSpan = document.getElementById('displayLat');
    const displayLngSpan = document.getElementById('displayLng');

    let allVideosData = [];
    let addFormMap = null;
    let addFormMarker = null;

    // --- 工具函式 ---

    function logSystemMessage(message, isError = false) {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('zh-TW', { hour12: false });
        const prefix = isError ? '[ERROR]' : '[INFO]';
        // 終端機風格符號
        const promptChar = isError ? '!' : '>'; 
        const fullMsg = `${timeStr} ${prefix} ${promptChar} ${message}\n`;
        
        aiResponseTextArea.value += fullMsg;
        aiResponseTextArea.scrollTop = aiResponseTextArea.scrollHeight;
        if (isError) console.error(message);
    }

    // 數字跳動動畫
    function animateValue(obj, start, end, duration) {
        if(!obj) return;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // 計算安全分數 (與 video_detail.js 邏輯保持一致)
    function calculateScore(dangerCount) {
        let score = 100 - (dangerCount * 3);
        if (score < 40) score = 40;
        return score;
    }

    // --- 核心邏輯 ---

    async function fetchAndRenderVideos() {
        videoPlayerArea.innerHTML = `<div style="width:100%; text-align:center; padding:40px; color:#666;"><p>⏳ 連線資料庫中...</p></div>`;
            
        try {
            const response = await fetch(`${API_BASE_URL}/get_videos`);
            if (!response.ok) throw new Error(`Status: ${response.status}`);
            allVideosData = await response.json();
            
            logSystemMessage(`資料庫同步完成，共 ${allVideosData.length} 筆資料。`);
            
            // 1. 更新上方總覽數據
            updateGlobalStats(allVideosData);
            
            // 2. 渲染影片列表
            renderVideoGallery(allVideosData);
            
        } catch (error) {
            videoPlayerArea.innerHTML = `<div style="width:100%; text-align:center; color:#dc3545;">⚠️ 載入失敗: ${error.message}</div>`;
            logSystemMessage(`載入失敗: ${error.message}`, true);
        }
    }

    function updateGlobalStats(videos) {
        let totalDangers = 0;
        let totalScoreSum = 0;
        
        videos.forEach(v => {
            const dCount = (v.danger_events || []).length;
            totalDangers += dCount;
            totalScoreSum += calculateScore(dCount);
        });

        const avgScore = videos.length > 0 ? Math.round(totalScoreSum / videos.length) : 0;

        animateValue(totalTripsEl, 0, videos.length, 1000);
        animateValue(totalDangersEl, 0, totalDangers, 1500);
        animateValue(totalAvgScoreEl, 0, avgScore, 1000);
        
        // 根據平均分改變顏色
        if (avgScore >= 80) totalAvgScoreEl.style.color = '#28a745';
        else if (avgScore >= 60) totalAvgScoreEl.style.color = '#ffc107';
        else totalAvgScoreEl.style.color = '#dc3545';
    }

    function renderVideoGallery(videos) {
        videoPlayerArea.innerHTML = '';
        
        if (!videos || videos.length === 0) {
            videoPlayerArea.innerHTML = `
                <div style="width:100%; text-align:center; padding:50px; color:#888; background:#f8f9fa; border-radius:8px;">
                    <h3>📭 尚無行程記錄</h3>
                    <p>開始您的第一趟 AI 智慧旅程吧！</p>
                </div>`;
            return;
        }

        videos.forEach(video => {
            // 計算該影片分數
            const dCount = (video.danger_events || []).length;
            const score = calculateScore(dCount);
            let badgeClass = 'score-high';
            if (score < 60) badgeClass = 'score-low';
            else if (score < 80) badgeClass = 'score-mid';

            const videoItem = document.createElement('div');
            videoItem.classList.add('video-item');
            videoItem.style.animation = "fadeIn 0.5s ease-out";
            
            videoItem.innerHTML = `
                <div class="video-item-header">
                    <h4 title="${video.title}">${video.title || 'Trip'}</h4>
                    <span style="font-size:0.8em; color:#888;">${video.date || ''}</span>
                    <button class="btn-delete-video" data-video-id="${video.id}" title="刪除">×</button>
                </div>
                
                <div class="video-content-wrapper">
                    <!-- 【新增】分數標籤 -->
                    <div class="score-badge ${badgeClass}">${score}分</div>
                    
                    ${video.content || '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#666;">無預覽</div>'}
                </div>
                
                <p class="video-description">
                    ${video.description ? video.description.substring(0, 30) + (video.description.length > 30 ? '...' : '') : '無描述'}
                </p>
            `;

            videoItem.addEventListener('click', (e) => {
                if (!e.target.closest('.btn-delete-video')) {
                    window.location.href = `video_detail.html?id=${video.id}`;
                }
            });

            const deleteButton = videoItem.querySelector('.btn-delete-video');
            deleteButton.addEventListener('click', async (e) => {
                e.stopPropagation();
                if (!confirm(`確定要刪除 "${video.title}" 嗎？`)) return;

                logSystemMessage(`正在刪除影片 ${video.id}...`);
                videoItem.style.opacity = '0.5';
                videoItem.style.pointerEvents = 'none';

                try {
                    const response = await fetch(`${API_BASE_URL}/delete_video/${video.id}`, { method: 'DELETE' });
                    if (!response.ok) throw new Error("刪除失敗");
                    
                    logSystemMessage(`影片已刪除。`);
                    videoItem.remove();
                    
                    // 重新計算總覽數據
                    allVideosData = allVideosData.filter(v => v.id !== video.id);
                    updateGlobalStats(allVideosData);
                    if (allVideosData.length === 0) renderVideoGallery([]); 

                } catch (error) {
                    videoItem.style.opacity = '1';
                    videoItem.style.pointerEvents = 'auto';
                    logSystemMessage(`刪除失敗: ${error.message}`, true);
                }
            });

            videoPlayerArea.appendChild(videoItem);
        });
    }

    // 搜尋功能
    document.getElementById('searchInput').addEventListener('input', (e) => {
        const term = e.target.value.trim().toLowerCase();
        const filtered = term ? allVideosData.filter(v =>
            (v.title && v.title.toLowerCase().includes(term)) ||
            (v.description && v.description.toLowerCase().includes(term))
        ) : allVideosData;
        renderVideoGallery(filtered);
    });

    // --- 檔案選擇與 Demo 功能 ---
    const localVideoPathGroup = document.getElementById('localVideoPathGroup');
    const videoPathInput = document.getElementById('videoPath');
    const videoTitleInput = document.getElementById('videoTitle'); // 順便自動填標題

    if (localVideoPathGroup && videoPathInput) {
        // 建立檔案選擇器
        const filePicker = document.createElement('input');
        filePicker.type = 'file';
        filePicker.accept = 'video/*';
        filePicker.style.marginBottom = '10px';

        // 建立一個進度條或提示訊息
        const uploadStatus = document.createElement('span');
        uploadStatus.style.marginLeft = '10px';
        uploadStatus.style.fontSize = '0.9em';
        uploadStatus.style.color = '#666';

        // 插入到 DOM
        localVideoPathGroup.insertBefore(filePicker, videoPathInput);
        localVideoPathGroup.insertBefore(uploadStatus, videoPathInput);

        // 監聽檔案選擇事件
        filePicker.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            // 1. 更新 UI 狀態
            uploadStatus.textContent = "⏳ 正在上傳影片到伺服器...";
            uploadStatus.style.color = '#e67e22'; // 橘色
            filePicker.disabled = true; // 鎖定不讓重複選

            // 2. 準備上傳資料
            const formData = new FormData();
            formData.append('video_file', file);

            try {
                // 3. 發送到後端 API
                const response = await fetch(`${API_BASE_URL}/upload_video_file`, {
                    method: 'POST',
                    body: formData // fetch 會自動處理 multipart/form-data
                });

                if (!response.ok) throw new Error('上傳失敗');

                const result = await response.json();

                // 4. 上傳成功：自動填入路徑
                videoPathInput.value = result.path; // 填入 videos/檔名.mp4

                // 如果標題還是空的，就自動幫他填檔名
                if (!videoTitleInput.value) {
                    videoTitleInput.value = result.filename;
                }

                uploadStatus.textContent = "✅ 上傳完成！";
                uploadStatus.style.color = '#27ae60'; // 綠色

            } catch (error) {
                console.error(error);
                uploadStatus.textContent = "❌ 上傳失敗，請重試。";
                uploadStatus.style.color = '#c0392b'; // 紅色
                videoPathInput.value = ""; // 清空錯誤路徑
            } finally {
                filePicker.disabled = false; // 解鎖
            }
        });
    }
    
    // Demo Mode Checkbox
    const formActions = document.querySelector('.form-actions');
    if (formActions && !document.getElementById('generateDemoData')) {
        const demoDiv = document.createElement('div');
        demoDiv.innerHTML = `<label style="cursor:pointer; color:#856404; background:#fff3cd; padding:8px; display:block; border-radius:4px; margin-bottom:10px;"><input type="checkbox" id="generateDemoData"> <b>生成測試數據 (Demo Mode)</b></label>`;
        formActions.parentNode.insertBefore(demoDiv, formActions);
    }

    // 表單提交
    addVideoForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        // ... (這部分邏輯與上一版相同，請確保保留您的 Demo 數據生成邏輯) ...
        // 為節省篇幅，請將您上一版 script.js 的 submit 邏輯複製過來，
        // 或者是直接使用您現在已經改好的 script.js，
        // 只需要把 renderVideoGallery 和 updateGlobalStats 這兩個函式換成我上面提供的版本即可。
        
        // 簡單來說：
        // 1. 取得表單資料
        // 2. 檢查 Demo Checkbox -> 生成 mockGpsTrace
        // 3. fetch POST
        // 4. 成功後 -> fetchAndRenderVideos()
        
        // 這裡為了讓您能直接複製使用，我寫一個標準的提交處理：
        const formData = new FormData(addVideoForm);
        const data = Object.fromEntries(formData.entries());
        let loc = {};
        if(data.lat_map_selected) loc = {lat: parseFloat(data.lat_map_selected), lng: parseFloat(data.lng_map_selected)};
        
        let mockGps = [], mockDanger = [];
        if(document.getElementById('generateDemoData')?.checked) {
            logSystemMessage("生成虛擬數據中...");
            // 簡單生成一點數據
            for(let i=0; i<10; i++) mockGps.push({lat: 25.03+i*0.001, lng: 121.56, timestamp: new Date().toISOString(), accuracy: 10});
            mockDanger.push({description: "測試警報", timestamp: new Date().toISOString()});
        }

        const payload = {
            title: data.title, date: data.date, description: data.description,
            content_type: 'local_video', path: data.path, location: loc,
            gps_trace: mockGps, danger_events: mockDanger
        };

        try {
            const res = await fetch(`${API_BASE_URL}/add_video`, {
                method: 'POST', 
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            if(!res.ok) throw new Error("新增失敗");
            logSystemMessage("新增成功！");
            toggleAddVideoForm(false);
            fetchAndRenderVideos();
        } catch(err) {
            logSystemMessage(err.message, true);
        }
    });

    // 切換表單顯示
    function toggleAddVideoForm(show) {
        if(show) {
            addVideoForm.reset();
            addVideoFormContainer.classList.remove('hidden');
            initializeOrUpdateAddFormMap();
        } else {
            addVideoFormContainer.classList.add('hidden');
        }
    }
    showAddVideoButton.addEventListener('click', () => toggleAddVideoForm(true));
    cancelAddVideoButton.addEventListener('click', () => toggleAddVideoForm(false));
    
    // 初始化地圖 (簡化版，請保留您原本完整的地圖邏輯)
    // 初始化或更新地圖 (修復版：加入點擊偵測與標記)
    function initializeOrUpdateAddFormMap() {
        // 1. 檢查 Google Maps API 是否載入完成
        if (typeof google === 'undefined' || !google.maps) {
            console.warn("Google Maps API 尚未就緒");
            return;
        }

        const mapElement = document.getElementById('addVideoMap');
        if (!mapElement) return;

        // 2. 如果地圖還沒建立，就建立它
        if (!addFormMap) {
            addFormMap = new google.maps.Map(mapElement, {
                center: { lat: 25.033, lng: 121.565 }, // 預設台北
                zoom: 12,
                mapTypeId: 'roadmap',
                clickableIcons: false, // 關閉地標點擊，避免干擾選點
                streetViewControl: false
            });

            // 🔥 關鍵修復：加入點擊監聽器 🔥
            addFormMap.addListener('click', (e) => {
                const lat = e.latLng.lat();
                const lng = e.latLng.lng();

                // (A) 更新隱藏欄位 (給後端用的)
                if(selectedLatInput) selectedLatInput.value = lat;
                if(selectedLngInput) selectedLngInput.value = lng;

                // (B) 更新畫面文字 (給使用者看的)
                if(displayLatSpan) displayLatSpan.textContent = lat.toFixed(6);
                if(displayLngSpan) displayLngSpan.textContent = lng.toFixed(6);

                // (C) 在地圖上插一根紅色的針 (Marker)
                placeMarker({ lat: lat, lng: lng });
            });
        }

        // 確保地圖在顯示時重新調整大小 (避免地圖只顯示一角)
        setTimeout(() => {
             google.maps.event.trigger(addFormMap, "resize");
             if(addFormMarker) {
                 addFormMap.setCenter(addFormMarker.getPosition());
             } else {
                 addFormMap.setCenter({ lat: 25.033, lng: 121.565 });
             }
        }, 100);
    }

    // 輔助函式：放置標記 (如果已有標記，就移動它)
    function placeMarker(location) {
        if (addFormMarker) {
            // 如果針已經存在，就移動位置
            addFormMarker.setPosition(location);
        } else {
            // 如果針不存在，就插一根新的
            addFormMarker = new google.maps.Marker({
                position: location,
                map: addFormMap,
                animation: google.maps.Animation.DROP, // 掉落動畫
                draggable: true // 允許拖拉微調
            });

            // 允許使用者拖拉標記來修正位置
            addFormMarker.addListener('dragend', (e) => {
                const lat = e.latLng.lat();
                const lng = e.latLng.lng();
                if(selectedLatInput) selectedLatInput.value = lat;
                if(selectedLngInput) selectedLngInput.value = lng;
                if(displayLatSpan) displayLatSpan.textContent = lat.toFixed(6);
                if(displayLngSpan) displayLngSpan.textContent = lng.toFixed(6);
            });
        }
    }

    // 啟動
    fetchAndRenderVideos();
});