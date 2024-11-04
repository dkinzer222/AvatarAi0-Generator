import numpy as np
from collections import deque
import time

class GestureRecognizer:
    def __init__(self, history_length=30):
        self.history_length = history_length
        self.landmark_history = deque(maxlen=history_length)
        self.last_gesture_time = {}  # Cooldown for gesture detection
        self.gesture_cooldown = 2.0  # Seconds between same gesture detection
        
    def add_landmarks(self, landmarks):
        """Add landmarks to history"""
        if landmarks is not None:
            self.landmark_history.append(landmarks)
            
    def detect_gestures(self):
        """Detect various gestures from landmark history"""
        if len(self.landmark_history) < self.history_length:
            return None
            
        current_time = time.time()
        detected_gestures = []
        
        # Check each gesture
        gestures = {
            'waving': self._detect_waving,
            'pointing': self._detect_pointing,
            'clapping': self._detect_clapping,
            'raising_hand': self._detect_raising_hand
        }
        
        for gesture_name, detect_func in gestures.items():
            # Check cooldown
            if current_time - self.last_gesture_time.get(gesture_name, 0) > self.gesture_cooldown:
                if detect_func():
                    detected_gestures.append(gesture_name)
                    self.last_gesture_time[gesture_name] = current_time
                    
        return detected_gestures[0] if detected_gestures else None
        
    def _detect_waving(self):
        """Detect waving gesture (side-to-side hand movement)"""
        # Use wrist points (left: 15, right: 16)
        wrist_indices = [15, 16]  # Left and right wrist
        min_wave_amplitude = 0.15  # Minimum wave movement
        min_cycles = 2  # Minimum number of back-and-forth movements
        
        for wrist_idx in wrist_indices:
            positions = np.array([frame[wrist_idx][:2] for frame in self.landmark_history])
            x_positions = positions[:, 0]
            
            # Calculate zero crossings of x-position around mean
            mean_x = np.mean(x_positions)
            zero_crossings = np.where(np.diff(np.signbit(x_positions - mean_x)))[0]
            
            amplitude = np.max(x_positions) - np.min(x_positions)
            
            if len(zero_crossings) >= min_cycles * 2 and amplitude > min_wave_amplitude:
                return True
                
        return False
        
    def _detect_pointing(self):
        """Detect pointing gesture (extended arm with index finger)"""
        # Check last frame for pointing pose
        current_frame = self.landmark_history[-1]
        
        # Check both arms
        for (shoulder, elbow, wrist) in [(11, 13, 15), (12, 14, 16)]:
            # Get joint positions
            shoulder_pos = current_frame[shoulder][:3]
            elbow_pos = current_frame[elbow][:3]
            wrist_pos = current_frame[wrist][:3]
            
            # Calculate angles
            vec1 = elbow_pos - shoulder_pos
            vec2 = wrist_pos - elbow_pos
            
            # Check if arm is extended (angle close to 180 degrees)
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            if angle > 2.8:  # About 160 degrees
                return True
                
        return False
        
    def _detect_clapping(self):
        """Detect clapping gesture (hands coming together repeatedly)"""
        # Use wrist points
        left_wrist_idx, right_wrist_idx = 15, 16
        min_clap_speed = 0.1
        min_claps = 2
        
        # Calculate distances between hands over time
        distances = []
        for frame in self.landmark_history:
            left_pos = frame[left_wrist_idx][:3]
            right_pos = frame[right_wrist_idx][:3]
            distance = np.linalg.norm(left_pos - right_pos)
            distances.append(distance)
            
        # Look for multiple distance minima (claps)
        distances = np.array(distances)
        min_indices = []
        for i in range(1, len(distances) - 1):
            if distances[i] < distances[i-1] and distances[i] < distances[i+1]:
                min_indices.append(i)
                
        if len(min_indices) >= min_claps:
            # Check if claps are fast enough
            time_between_claps = len(distances) / len(min_indices)
            return time_between_claps < (self.history_length / min_clap_speed)
            
        return False
        
    def _detect_raising_hand(self):
        """Detect raised hand gesture"""
        # Use wrist and shoulder points
        for wrist_idx, shoulder_idx in [(15, 11), (16, 12)]:  # Check both arms
            current_frame = self.landmark_history[-1]
            wrist_pos = current_frame[wrist_idx]
            shoulder_pos = current_frame[shoulder_idx]
            
            # Check if wrist is significantly above shoulder
            if wrist_pos[1] < shoulder_pos[1] - 0.2:  # Y coordinates, lower value means higher position
                return True
                
        return False
