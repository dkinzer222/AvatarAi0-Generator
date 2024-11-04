let poseDetection;

async function setupPoseDetection() {
    const pose = new Pose({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
        }
    });

    pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    pose.onResults(onPoseResults);

    // Setup webcam
    const video = document.createElement('video');
    video.setAttribute('playsinline', '');
    
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    });
    video.srcObject = stream;
    video.play();

    // Start detection loop
    async function detect() {
        await pose.send({image: video});
        requestAnimationFrame(detect);
    }

    detect();
}

function onPoseResults(results) {
    if (results.poseLandmarks) {
        socket.emit('pose_data', results.poseLandmarks);
    }
}

window.addEventListener('load', setupPoseDetection);
