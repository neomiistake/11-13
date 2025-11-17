// live_view.js

document.addEventListener('DOMContentLoaded', () => {
    // --- 配置 ---
    // 【重要】請確保這個 IP 是您樹莓派 MediaMTX 伺服器的正確 IP
    const mediamtxBaseUrl = "http://192.168.196.73:8889";
    const streamPaths = ["cam0", "cam1"];

    // 異步函式，用於連接單個 WebRTC 串流
    async function connectStream(path) {
        const videoElement = document.getElementById(`video-${path}`);
        const statusElement = document.getElementById(`status-${path}`);

        if (!videoElement || !statusElement) {
            console.error(`找不到 ${path} 的 HTML 元素`);
            return;
        }

        const pc = new RTCPeerConnection();

        pc.oniceconnectionstatechange = () => {
            statusElement.textContent = `狀態: ${pc.iceConnectionState}`;
            if (pc.iceConnectionState === 'connected' || pc.iceConnectionState === 'completed') {
                 statusElement.style.opacity = 0; // 連線成功後淡出狀態
            } else {
                 statusElement.style.opacity = 1;
            }
        };

        pc.ontrack = (event) => {
            if (event.track.kind === 'video') {
                videoElement.srcObject = event.streams[0];
            }
        };

        try {
            pc.addTransceiver('video', { 'direction': 'recvonly' });

            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            const response = await fetch(`${mediamtxBaseUrl}/${path}/whep`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/sdp' },
                body: offer.sdp
            });

            if (response.status !== 201) {
                throw new Error(`伺服器回應錯誤: ${response.status}`);
            }

            const answerSdp = await response.text();
            await pc.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp: answerSdp }));

        } catch (error) {
            console.error(`連接到 ${path} 失敗:`, error);
            statusElement.textContent = `錯誤: 連接失敗`;
            statusElement.style.backgroundColor = 'rgba(220, 53, 69, 0.8)';
        }
    }

    // 遍歷所有串流路徑並啟動連接
    streamPaths.forEach(path => {
        connectStream(path);
    });
});