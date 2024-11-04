from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import time
from utils.pose_tracker import PoseTracker
from utils.smplx_renderer import SMPLXRenderer
from utils.calibration import CalibrationGuide

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# Initialize components
pose_tracker = PoseTracker()
avatar_renderer = SMPLXRenderer()
calibration_guide = CalibrationGuide()

# Track processing times for progress indicators
processing_stats = {
    'pose_detection': {'start': 0, 'duration': 0},
    'avatar_rendering': {'start': 0, 'duration': 0}
}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('calibration_instruction', calibration_guide.get_current_instruction())

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

@socketio.on('start_calibration')
def handle_start_calibration():
    """Handle calibration start request"""
    instruction = calibration_guide.start_calibration()
    emit('calibration_instruction', instruction)

@socketio.on('update_avatar')
def handle_avatar_update(data):
    try:
        # Convert color from hex to RGB
        if 'color' in data:
            hex_color = data['color'].lstrip('#')
            rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            avatar_renderer.set_customization(color=rgb_color)
            
        # Update other customization options
        if 'size' in data:
            avatar_renderer.set_customization(size=float(data['size']))
        if 'style' in data:
            avatar_renderer.set_customization(style=data['style'])
        if 'lineThickness' in data:
            avatar_renderer.set_customization(line_thickness=int(data['lineThickness']))
        if 'jointSize' in data:
            avatar_renderer.set_customization(joint_size=float(data['jointSize']))
            
        emit('customization_updated', {'status': 'success'})
    except Exception as e:
        print(f"Error updating avatar: {str(e)}")
        emit('customization_updated', {'status': 'error', 'message': str(e)})

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # Start pose detection timing
        processing_stats['pose_detection']['start'] = time.time()
        
        # Decode base64 image
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None and frame.size > 0:
            # Process frame with MediaPipe
            landmarks, face_landmarks, expression, gesture, processed_frame = pose_tracker.process_frame(frame)
            
            # Update pose detection timing
            processing_stats['pose_detection']['duration'] = time.time() - processing_stats['pose_detection']['start']
            
            # Start avatar rendering timing
            processing_stats['avatar_rendering']['start'] = time.time()
            
            if landmarks is not None:
                # Update calibration state if landmarks detected
                calibration_instruction = calibration_guide.update_calibration(landmarks)
                emit('calibration_instruction', calibration_instruction)
                
                # Draw pose landmarks, facial expression, and gestures on original frame
                pose_frame = pose_tracker.draw_pose(frame.copy(), landmarks, face_landmarks, expression, gesture)
                
                # Render SMPL-X avatar
                avatar_frame = avatar_renderer.render_avatar(landmarks, expression)
                
                # Update avatar rendering timing
                processing_stats['avatar_rendering']['duration'] = time.time() - processing_stats['avatar_rendering']['start']
                
                # Convert frames to base64 for sending back to client
                _, buffer = cv2.imencode('.jpg', pose_frame)
                pose_data = base64.b64encode(buffer).decode('utf-8')
                
                _, buffer = cv2.imencode('.jpg', avatar_frame)
                avatar_data = base64.b64encode(buffer).decode('utf-8')
                
                # Calculate processing progress percentages
                pose_progress = min(100, (processing_stats['pose_detection']['duration'] / 0.033) * 100)
                avatar_progress = min(100, (processing_stats['avatar_rendering']['duration'] / 0.033) * 100)
                
                # Send both frames back to client along with expression, gesture, and progress
                emit('processed_frame', {
                    'pose_frame': f'data:image/jpeg;base64,{pose_data}',
                    'avatar_frame': f'data:image/jpeg;base64,{avatar_data}',
                    'expression': expression,
                    'gesture': gesture,
                    'processing_progress': {
                        'pose_detection': pose_progress,
                        'avatar_rendering': avatar_progress
                    }
                })
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        emit('error', {'message': str(e)})
