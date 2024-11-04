const socket = io();

socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('avatar_update', (data) => {
    updateAvatarPose(data);
});

socket.on('voice_response', (data) => {
    console.log('Voice response:', data);
});

// Handle audio recording and voice commands
let mediaRecorder;
let audioChunks = [];

async function setupVoiceRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            socket.emit('voice_command', audioBlob);
            audioChunks = [];
        };
    } catch (error) {
        console.error('Error accessing microphone:', error);
    }
}

window.addEventListener('load', setupVoiceRecording);
