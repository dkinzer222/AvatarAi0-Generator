from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from utils.pose_tracker import PoseTracker
from utils.smplx_renderer import SMPLXRenderer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# Initialize pose tracker and avatar renderer
pose_tracker = PoseTracker()
avatar_renderer = SMPLXRenderer()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # Decode base64 image
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None and frame.size > 0:
            # Process frame with MediaPipe
            landmarks, face_landmarks, expression, gesture, processed_frame = pose_tracker.process_frame(frame)
            
            if landmarks is not None:
                # Draw pose landmarks, facial expression, and gestures on original frame
                pose_frame = pose_tracker.draw_pose(frame.copy(), landmarks, face_landmarks, expression, gesture)
                
                # Render SMPL-X avatar
                avatar_frame = avatar_renderer.render_avatar(landmarks, expression)
                
                # Convert frames to base64 for sending back to client
                _, buffer = cv2.imencode('.jpg', pose_frame)
                pose_data = base64.b64encode(buffer).decode('utf-8')
                
                _, buffer = cv2.imencode('.jpg', avatar_frame)
                avatar_data = base64.b64encode(buffer).decode('utf-8')
                
                # Send both frames back to client along with expression and gesture
                emit('processed_frame', {
                    'pose_frame': f'data:image/jpeg;base64,{pose_data}',
                    'avatar_frame': f'data:image/jpeg;base64,{avatar_data}',
                    'expression': expression,
                    'gesture': gesture
                })
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
