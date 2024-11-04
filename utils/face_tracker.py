import mediapipe as mp
import numpy as np
import cv2

class FaceTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define facial expression landmarks for common expressions
        self.expression_landmarks = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'mouth': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
            'eyebrows': [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
        }
        
    def process_frame(self, frame):
        """Process a frame and return face landmarks"""
        if frame is None:
            return None
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process the frame
        results = self.face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True
        
        if not results.multi_face_landmarks:
            return None
            
        # Get the first face detected
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to numpy array
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
            
        return np.array(landmarks)
        
    def detect_expression(self, landmarks):
        """Detect facial expression based on landmark positions"""
        if landmarks is None:
            return "neutral"
            
        # Calculate basic metrics for expression detection
        expressions = {}
        
        # Eye opening ratio
        left_eye_ratio = self._calculate_eye_ratio(landmarks, self.expression_landmarks['left_eye'])
        right_eye_ratio = self._calculate_eye_ratio(landmarks, self.expression_landmarks['right_eye'])
        expressions['eyes'] = (left_eye_ratio + right_eye_ratio) / 2
        
        # Mouth opening ratio
        mouth_ratio = self._calculate_mouth_ratio(landmarks, self.expression_landmarks['mouth'])
        expressions['mouth'] = mouth_ratio
        
        # Eyebrow position
        eyebrow_position = self._calculate_eyebrow_position(landmarks, self.expression_landmarks['eyebrows'])
        expressions['eyebrows'] = eyebrow_position
        
        # Determine expression based on metrics
        if expressions['eyes'] < 0.2:
            return "closed_eyes"
        elif expressions['mouth'] > 0.5:
            return "open_mouth"
        elif expressions['eyebrows'] > 0.6:
            return "raised_eyebrows"
        elif expressions['eyebrows'] < 0.3:
            return "frown"
            
        return "neutral"
        
    def _calculate_eye_ratio(self, landmarks, eye_indices):
        """Calculate eye opening ratio"""
        points = landmarks[eye_indices]
        vertical_dist = np.mean([
            np.linalg.norm(points[1] - points[5]),
            np.linalg.norm(points[2] - points[4])
        ])
        horizontal_dist = np.linalg.norm(points[0] - points[3])
        return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
        
    def _calculate_mouth_ratio(self, landmarks, mouth_indices):
        """Calculate mouth opening ratio"""
        points = landmarks[mouth_indices]
        vertical_dist = np.linalg.norm(points[3] - points[9])
        horizontal_dist = np.linalg.norm(points[0] - points[6])
        return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
        
    def _calculate_eyebrow_position(self, landmarks, eyebrow_indices):
        """Calculate relative eyebrow position"""
        points = landmarks[eyebrow_indices]
        return np.mean([p[1] for p in points])  # Use y-coordinate
