from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import time
import logging

from utils.pose_tracker import PoseTracker
from utils.smplx_renderer import SMPLXRenderer
from utils.calibration import CalibrationGuide

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_routes(app, socketio):
    pose_tracker = PoseTracker()
    avatar_renderer = SMPLXRenderer()
    calibration_guide = CalibrationGuide()

    # Track processing times for progress indicators
    processing_stats = {
        'pose_detection': {'start': 0, 'duration': 0},
        'avatar_rendering': {'start': 0, 'duration': 0}
    }

    def resize_frame_for_mobile(frame, max_dimension=640):
        """Resize frame while maintaining aspect ratio and quality"""
        if frame is None:
            return None
            
        try:
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                return None

            # Calculate new dimensions while maintaining aspect ratio
            aspect_ratio = width / height
            if height > width:
                if height > max_dimension:
                    new_height = max_dimension
                    new_width = int(max_dimension * aspect_ratio)
            else:
                if width > max_dimension:
                    new_width = max_dimension
                    new_height = int(max_dimension / aspect_ratio)
            
            # Only resize if necessary
            if height > max_dimension or width > max_dimension:
                frame = cv2.resize(frame, (new_width, new_height), 
                                interpolation=cv2.INTER_AREA)
            
            # Ensure proper color space and orientation
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                
            return frame
        except Exception as e:
            logger.error(f"Error resizing frame: {str(e)}")
            return None

    def optimize_frame_for_mobile(frame, quality=85):
        """Optimize frame for mobile transmission"""
        try:
            if frame is None:
                return None
            
            # Convert to RGB if necessary
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                
            # Apply moderate compression with quality setting
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, quality,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                cv2.IMWRITE_JPEG_PROGRESSIVE, 1
            ]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if not success:
                raise ValueError("Failed to encode frame")
                
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Error optimizing frame: {str(e)}")
            return None

    @app.route('/')
    def index():
        user_agent = request.headers.get('User-Agent', '').lower()
        is_mobile = any(device in user_agent for device in ['mobile', 'android', 'iphone', 'ipad', 'ipod'])
        return render_template('index.html', is_mobile=is_mobile)

    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('calibration_instruction', calibration_guide.get_current_instruction())

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")

    @socketio.on('start_calibration')
    def handle_start_calibration():
        instruction = calibration_guide.start_calibration()
        emit('calibration_instruction', instruction)

    @socketio.on('update_avatar')
    def handle_avatar_update(data):
        try:
            if 'color' in data:
                hex_color = data['color'].lstrip('#')
                rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                avatar_renderer.set_customization(color=rgb_color)
                
            for key, attr in {
                'size': float, 
                'style': str,
                'lineThickness': int,
                'jointSize': float
            }.items():
                if key in data:
                    try:
                        value = attr(data[key])
                        avatar_renderer.set_customization(**{key.lower(): value})
                    except (ValueError, TypeError) as e:
                        logger.error(f"Invalid value for {key}: {str(e)}")
                        emit('error', {'message': f'Invalid value for {key}'})
                        return
                        
            emit('customization_updated', {'status': 'success'})
        except Exception as e:
            logger.error(f"Error updating avatar: {str(e)}")
            emit('error', {'message': 'Error updating avatar customization'})

    @socketio.on('video_frame')
    def handle_video_frame(data):
        try:
            processing_stats['pose_detection']['start'] = time.time()
            
            try:
                # Handle both data URL format and raw base64
                encoded_data = data.split(',')[1] if ',' in data else data
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    raise ValueError("Failed to decode image data")
                    
            except Exception as e:
                logger.error(f"Error decoding frame: {str(e)}")
                emit('error', {'message': 'Invalid video frame data'})
                return

            if frame.size > 0:
                # Optimize frame for mobile
                frame = resize_frame_for_mobile(frame)
                if frame is None:
                    emit('error', {'message': 'Error processing video frame'})
                    return
                
                try:
                    # Process frame with MediaPipe
                    landmarks, face_landmarks, expression, gesture, processed_frame = pose_tracker.process_frame(frame)
                    processing_stats['pose_detection']['duration'] = time.time() - processing_stats['pose_detection']['start']
                    
                    if landmarks is not None:
                        # Update calibration state
                        calibration_instruction = calibration_guide.update_calibration(landmarks)
                        emit('calibration_instruction', calibration_instruction)
                        
                        # Start avatar rendering with timing
                        processing_stats['avatar_rendering']['start'] = time.time()
                        pose_frame = pose_tracker.draw_pose(frame.copy(), landmarks, face_landmarks, expression, gesture)
                        avatar_frame = avatar_renderer.render_avatar(landmarks, expression)
                        processing_stats['avatar_rendering']['duration'] = time.time() - processing_stats['avatar_rendering']['start']
                        
                        # Optimize frames for mobile
                        pose_data = optimize_frame_for_mobile(pose_frame)
                        avatar_data = optimize_frame_for_mobile(avatar_frame)
                        
                        if pose_data and avatar_data:
                            # Calculate and normalize processing progress
                            pose_progress = min(100, (processing_stats['pose_detection']['duration'] / 0.033) * 100)
                            avatar_progress = min(100, (processing_stats['avatar_rendering']['duration'] / 0.033) * 100)
                            
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
                        else:
                            emit('error', {'message': 'Error optimizing frames for mobile'})
                    else:
                        emit('error', {'message': 'No pose detected'})
                        
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    emit('error', {'message': 'Error processing video frame'})
                    
        except Exception as e:
            logger.error(f"Error in video frame handler: {str(e)}")
            emit('error', {'message': 'Internal server error'})

    return app
