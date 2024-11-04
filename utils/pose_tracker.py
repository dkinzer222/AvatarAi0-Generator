import cv2
import mediapipe as mp
import numpy as np
from .face_tracker import FaceTracker
from .gesture_recognizer import GestureRecognizer

class PoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_tracker = FaceTracker()
        self.gesture_recognizer = GestureRecognizer()
        
        # Configure pose tracking for CPU operation
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_frame(self, frame):
        """Process a frame and return pose landmarks, face expression, and detected gestures"""
        if frame is None or frame.size == 0:
            return None, None, None, None, None
            
        try:
            # Get image dimensions
            height, width = frame.shape[:2]
                
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            image_rgb.flags.writeable = False
            pose_results = self.pose.process(image_rgb)
            
            # Process face landmarks
            face_landmarks = self.face_tracker.process_frame(frame)
            expression = self.face_tracker.detect_expression(face_landmarks) if face_landmarks is not None else None
            
            image_rgb.flags.writeable = True
            
            if pose_results.pose_landmarks:
                # Convert landmarks to numpy array with better normalization
                landmarks = []
                for landmark in pose_results.pose_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z * width
                    visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    
                    landmarks.append([x, y, z, visibility])
                
                landmarks = np.array(landmarks)
                landmarks[:, :2] = np.clip(landmarks[:, :2], 0, 1)
                
                # Update gesture recognizer and detect gestures
                self.gesture_recognizer.add_landmarks(landmarks)
                gesture = self.gesture_recognizer.detect_gestures()
                
                return landmarks, face_landmarks, expression, gesture, image_rgb
                
            return None, face_landmarks, expression, None, image_rgb
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None, None, None, None, None
        
    def draw_pose(self, image, landmarks, face_landmarks=None, expression=None, gesture=None):
        """Draw pose landmarks, facial expression, and detected gestures on the image"""
        if landmarks is None or image is None:
            return image
            
        try:
            # Convert BGR to RGB for consistency
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Draw pose landmarks
            height, width = image_rgb.shape[:2]
            for i, (x, y, z, v) in enumerate(landmarks):
                if v > 0.5:
                    px = int(x * width)
                    py = int(y * height)
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(image_rgb, (px, py), 5, (0, 255, 0), -1)
                    
            # Draw connections
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                    landmarks[start_idx][3] > 0.5 and landmarks[end_idx][3] > 0.5):
                    
                    start_point = (int(landmarks[start_idx][0] * width),
                                int(landmarks[start_idx][1] * height))
                    end_point = (int(landmarks[end_idx][0] * width),
                                int(landmarks[end_idx][1] * height))
                    
                    if (0 <= start_point[0] < width and 0 <= start_point[1] < height and
                        0 <= end_point[0] < width and 0 <= end_point[1] < height):
                        cv2.line(image_rgb, start_point, end_point, (255, 0, 0), 2)
            
            # Draw facial expression if available
            text_y = 30
            if expression:
                cv2.putText(image_rgb, f"Expression: {expression}", 
                           (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 255), 2)
                text_y += 40
                
            # Draw detected gesture if available
            if gesture:
                cv2.putText(image_rgb, f"Gesture: {gesture}", 
                           (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (255, 255, 0), 2)
            
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error drawing pose: {str(e)}")
            return image
