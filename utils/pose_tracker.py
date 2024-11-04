import cv2
import mediapipe as mp
import numpy as np

class PoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure pose tracking for CPU operation
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Using medium complexity for better performance
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_frame(self, frame):
        """Process a frame and return pose landmarks"""
        if frame is None or frame.size == 0:
            return None, None
            
        try:
            # Get image dimensions
            height, width = frame.shape[:2]
                
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)
            image_rgb.flags.writeable = True
            
            if results.pose_landmarks:
                # Convert landmarks to numpy array with better normalization
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    # Normalize coordinates to image dimensions
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z * width  # Scale Z coordinate relative to image width
                    visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    
                    landmarks.append([x, y, z, visibility])
                
                landmarks = np.array(landmarks)
                
                # Ensure landmarks are within bounds
                landmarks[:, :2] = np.clip(landmarks[:, :2], 0, 1)
                
                return landmarks, image_rgb
                
            return None, image_rgb
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None, None
        
    def draw_pose(self, image, landmarks):
        """Draw pose landmarks on the image"""
        if landmarks is None or image is None:
            return image
            
        try:
            # Convert BGR to RGB for consistency
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Draw landmarks manually with error checking
            height, width = image_rgb.shape[:2]
            for i, (x, y, z, v) in enumerate(landmarks):
                if v > 0.5:  # Only draw visible landmarks
                    px = int(x * width)
                    py = int(y * height)
                    # Ensure points are within image bounds
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(image_rgb, (px, py), 5, (0, 255, 0), -1)
                    
            # Draw connections with bounds checking
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                    landmarks[start_idx][3] > 0.5 and landmarks[end_idx][3] > 0.5):
                    
                    start_point = (int(landmarks[start_idx][0] * width),
                                int(landmarks[start_idx][1] * height))
                    end_point = (int(landmarks[end_idx][0] * width),
                                int(landmarks[end_idx][1] * height))
                    
                    # Check if points are within image bounds
                    if (0 <= start_point[0] < width and 0 <= start_point[1] < height and
                        0 <= end_point[0] < width and 0 <= end_point[1] < height):
                        cv2.line(image_rgb, start_point, end_point, (255, 0, 0), 2)
            
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error drawing pose: {str(e)}")
            return image
