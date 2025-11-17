// script.js (最終版)

window.initAppMaps = function() {
    console.log("Google Maps API ready for index.html.");
};

document.addEventListener('DOMContentLoaded', async () => {
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');
    const videoPlayerArea = document.getElementById('videoPlayerArea');
    const aiResponseTextArea = document.getElementById('aiResponseTextArea');
    const showAddVideoButton = document.getElementById('showAddVideoButton');
    const addVideoFormContainer = document.getElementById('addVideoFormContainer');
    const addVideoForm = document.getElementById('addVideoForm');
    const cancelAddVideoButton = document.getElementById('cancelAddVideoButton');
    const videoContentTypeSelect = document.getElementById('videoContentType');
    const localVideoPathGroup = document.getElementById('localVideoPathGroup');

    const addVideoMapDiv = document.getElementById('addVideoMap');
    const locationSearchInput = document.getElementById('locationSearchInput');
    const selectedLatInput = document.getElementById('selectedLat');
    const selectedLngInput = document.getElementById('selectedLng');
    const displayLatSpan = document.getElementById('displayLat');
    const displayLngSpan = document.getElementById('displayLng');

    let allVideosData = [];
    let addFormMap = null;
    let addFormMarker = null;

    function handleFetchError(error, actionType = "操作") {
        console.error(`執行 ${actionType} 時發生錯誤:`, error);
        aiResponseTextArea.value = `${actionType}失敗: ${error.message || "未知錯誤"}`;
    }

    async function fetchAndRenderVideos() {
        videoPlayerArea.innerHTML = '<p class="placeholder-text">影片載入中...</p>';
        try {
            const response = await fetch('http://127.0.0.1:5000/get_videos');
            if (!response.ok) throw new Error(`伺服器錯誤 ${response.status}`);
            allVideosData = await response.json();
            renderVideoGallery(allVideosData);
        } catch (error) {
            handleFetchError(error, "載入影片");
        }
    }

    function renderVideoGallery(videos) {
        videoPlayerArea.innerHTML = '';
        if (!videos || videos.length === 0) {
            videoPlayerArea.innerHTML = '<p class="placeholder-text">目前沒有影片。點擊 "+" 新增。</p>';
            return;
        }
        videos.forEach(video => {
            const videoItem = document.createElement('div');
            videoItem.classList.add('video-item');
            videoItem.innerHTML = `
                <div class="video-item-header">
                    <h4>${video.title || '無標題'} (${video.date || '未知'})</h4>
                    <button class="btn-delete-video" data-video-id="${video.id}" title="刪除">×</button>
                </div>
                <div class="video-content-wrapper">${video.content || '<p class="video-error">無法顯示</p>'}</div>
                <p class="video-description"><small>描述: ${video.description || '無'}</small></p>
            `;
            videoItem.addEventListener('click', (event) => {
                if (!event.target.closest('.btn-delete-video')) {
                    window.location.href = `video_detail.html?id=${video.id}`;
                }
            });
            const deleteButton = videoItem.querySelector('.btn-delete-video');
            deleteButton.addEventListener('click', async (e) => {
                e.stopPropagation();
                if (confirm(`確定要刪除影片 "${video.title || video.id}" 嗎？`)) {
                    aiResponseTextArea.value = `正在刪除影片 ${video.id}...`;
                    try {
                        const response = await fetch(`http://127.0.0.1:5000/delete_video/${video.id}`, { method: 'DELETE' });
                        if (!response.ok) {
                            const errData = await response.json();
                            throw new Error(errData.error || `伺服器錯誤 ${response.status}`);
                        }
                        const result = await response.json();
                        aiResponseTextArea.value = result.message;
                        fetchAndRenderVideos();
                    } catch (error) {
                        handleFetchError(error, `刪除影片 ${video.id}`);
                    }
                }
            });
            videoPlayerArea.appendChild(videoItem);
        });
    }

    function handleSearch() {
        const term = searchInput.value.trim().toLowerCase();
        const filtered = term ? allVideosData.filter(v =>
            (v.title && v.title.toLowerCase().includes(term)) ||
            (v.description && v.description.toLowerCase().includes(term))
        ) : allVideosData;
        renderVideoGallery(filtered);
    }
    searchButton.addEventListener('click', handleSearch);
    searchInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') handleSearch(); });

    function initializeOrUpdateAddFormMap(lat = 25.033976, lng = 121.564500) {
        if (typeof google === 'undefined' || !google.maps || !google.maps.places) {
            addVideoMapDiv.innerHTML = '<p class="map-error">地圖服務無法載入</p>';
            return;
        }
        const center = { lat, lng };
        if (!addFormMap) {
            addFormMap = new google.maps.Map(addVideoMapDiv, { center, zoom: 12, gestureHandling: 'cooperative' });
            addFormMarker = new google.maps.Marker({ position: center, map: addFormMap, draggable: true });
            addFormMap.addListener('click', (e) => { addFormMarker.setPosition(e.latLng); updateSelectedLocationUI(e.latLng); });
            addFormMarker.addListener('dragend', () => updateSelectedLocationUI(addFormMarker.getPosition()));
            const autocomplete = new google.maps.places.Autocomplete(locationSearchInput, { fields: ["geometry"] });
            autocomplete.bindTo("bounds", addFormMap);
            autocomplete.addListener("place_changed", () => {
                const place = autocomplete.getPlace();
                if (place.geometry && place.geometry.location) {
                    addFormMap.setCenter(place.geometry.location);
                    addFormMap.setZoom(17);
                    addFormMarker.setPosition(place.geometry.location);
                    updateSelectedLocationUI(place.geometry.location);
                }
            });
        } else {
            addFormMap.setCenter(center);
            addFormMarker.setPosition(center);
        }
        updateSelectedLocationUI(addFormMarker.getPosition());
    }

    function updateSelectedLocationUI(latLng) {
        const lat = latLng.lat();
        const lng = latLng.lng();
        selectedLatInput.value = lat.toFixed(6);
        selectedLngInput.value = lng.toFixed(6);
        displayLatSpan.textContent = lat.toFixed(6);
        displayLngSpan.textContent = lng.toFixed(6);
    }

    // 【動畫優化】
    function toggleAddVideoForm(show) {
        if (show) {
            addVideoForm.reset();
            displayLatSpan.textContent = "未選擇";
            displayLngSpan.textContent = "未選擇";
            selectedLatInput.value = "";
            selectedLngInput.value = "";

            addVideoFormContainer.classList.remove('hidden');

            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    (pos) => initializeOrUpdateAddFormMap(pos.coords.latitude, pos.coords.longitude),
                    () => initializeOrUpdateAddFormMap()
                );
            } else {
                initializeOrUpdateAddFormMap();
            }
        } else {
            addVideoFormContainer.classList.add('hidden');
        }
    }
    showAddVideoButton.addEventListener('click', () => toggleAddVideoForm(true));
    cancelAddVideoButton.addEventListener('click', () => toggleAddVideoForm(false));

    addVideoForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(addVideoForm);
        const videoData = Object.fromEntries(formData.entries());

        let locationToSend = {};
        const latMap = parseFloat(videoData.lat_map_selected);
        const lngMap = parseFloat(videoData.lng_map_selected);
        if (!isNaN(latMap) && !isNaN(lngMap)) {
            locationToSend = { lat: latMap, lng: lngMap };
        }

        const finalPayload = {
            title: videoData.title,
            date: videoData.date,
            description: videoData.description,
            content_type: 'local_video',
            path: videoData.path,
            location: locationToSend,
        };

        aiResponseTextArea.value = "正在新增影片...";
        try {
            const response = await fetch('http://127.0.0.1:5000/add_video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(finalPayload)
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || `伺服器錯誤 ${response.status}`);
            }
            const addedVideo = await response.json();
            aiResponseTextArea.value = `影片 "${addedVideo.title}" 新增成功！`;
            toggleAddVideoForm(false);
            fetchAndRenderVideos();
        } catch (error) {
            handleFetchError(error, "新增影片");
        }
    });

    fetchAndRenderVideos();
});