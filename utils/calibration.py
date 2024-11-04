import time
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CalibrationState(Enum):
    NOT_STARTED = "not_started"
    HEAD_TURN = "head_turn"
    ARMS_RAISE = "arms_raise"
    BODY_TURN = "body_turn" 
    SQUAT = "squat"
    COMPLETED = "completed"

class CalibrationGuide:
    def __init__(self):
        self.current_state = CalibrationState.NOT_STARTED
        self.state_start_time = None
        self.state_durations = {
            CalibrationState.HEAD_TURN: 8,
            CalibrationState.ARMS_RAISE: 8,
            CalibrationState.BODY_TURN: 8,
            CalibrationState.SQUAT: 8
        }
        # Enhanced movement thresholds for better accuracy
        self.movement_thresholds = {
            'head_rotation': 0.25,  # Reduced for easier detection
            'arm_raise': 0.35,      # Adjusted for better arm movement tracking
            'body_rotation': 0.2,   # Made more sensitive
            'squat_depth': 0.15     # Adjusted for better squat detection
        }
        self.movement_progress = {
            'left': 0,
            'right': 0,
            'up': 0,
            'down': 0
        }
        self.movement_history = []  # Track recent movements for smoothing
        self.history_size = 5       # Number of frames to keep for smoothing
        self.instructions = {
            CalibrationState.NOT_STARTED: {
                "title": "🎯 Let's Calibrate Your Avatar",
                "text": "Follow these simple steps for the best experience!",
                "details": [
                    "Find a well-lit space",
                    "Stand 2-3 meters from camera",
                    "Ensure full body visibility",
                    "Wear contrasting clothing"
                ],
                "success_criteria": "Click Start when ready!"
            },
            CalibrationState.HEAD_TURN: {
                "title": "👤 Head Movement",
                "text": "Let's start with some gentle head movements",
                "details": [
                    "1. Look straight ahead",
                    "2. Slowly turn head left",
                    "3. Return to center",
                    "4. Slowly turn head right",
                    "Keep shoulders still"
                ],
                "success_criteria": "Head movements detected successfully"
            },
            CalibrationState.ARMS_RAISE: {
                "title": "💪 Arm Movement",
                "text": "Next, let's check your arm range",
                "details": [
                    "1. Start with arms at sides",
                    "2. Raise both arms forward",
                    "3. Continue until overhead",
                    "4. Hold briefly at top",
                    "5. Lower arms slowly"
                ],
                "success_criteria": "Full arm range captured"
            },
            CalibrationState.BODY_TURN: {
                "title": "🔄 Body Rotation",
                "text": "Now for some gentle body turns",
                "details": [
                    "1. Face forward",
                    "2. Plant feet shoulder-width",
                    "3. Turn upper body left",
                    "4. Return to center",
                    "5. Turn upper body right"
                ],
                "success_criteria": "Body rotation range detected"
            },
            CalibrationState.SQUAT: {
                "title": "⬇️ Lower Body Check",
                "text": "Finally, a simple knee bend",
                "details": [
                    "1. Stand with feet apart",
                    "2. Keep back straight",
                    "3. Bend knees slightly",
                    "4. Hold position briefly",
                    "5. Return to standing"
                ],
                "success_criteria": "Lower body movement detected"
            },
            CalibrationState.COMPLETED: {
                "title": "✨ All Set!",
                "text": "Calibration complete - your avatar is ready!",
                "details": [
                    "All movements captured",
                    "Avatar personalized to you",
                    "Tracking optimized",
                    "Ready to begin!"
                ],
                "success_criteria": "Calibration successful!"
            }
        }
        
    def start_calibration(self):
        """Initialize calibration process with enhanced logging"""
        logger.info("Starting calibration process")
        self.current_state = CalibrationState.HEAD_TURN
        self.state_start_time = time.time()
        self.reset_movement_progress()
        self.movement_history = []
        return self.get_current_instruction()
        
    def reset_movement_progress(self):
        """Reset progress tracking with initialization logging"""
        logger.info(f"Resetting movement progress for state: {self.current_state}")
        self.movement_progress = {
            'left': 0,
            'right': 0,
            'up': 0,
            'down': 0
        }
        self.movement_history = []
        
    def _smooth_movement(self, new_value, direction):
        """Apply smoothing to movement detection"""
        self.movement_history.append({direction: new_value})
        if len(self.movement_history) > self.history_size:
            self.movement_history.pop(0)
            
        # Calculate smoothed value using weighted average
        weights = np.linspace(0.5, 1.0, len(self.movement_history))
        smoothed = 0
        weight_sum = 0
        
        for i, hist in enumerate(self.movement_history):
            if direction in hist:
                smoothed += hist[direction] * weights[i]
                weight_sum += weights[i]
                
        return smoothed / weight_sum if weight_sum > 0 else 0
        
    def update_calibration(self, landmarks):
        """Update calibration state with enhanced feedback"""
        if landmarks is None or self.current_state == CalibrationState.COMPLETED:
            return self.get_current_instruction()
            
        if self.state_start_time is None:
            return self.get_current_instruction()
            
        try:
            # Update movement progress with smoothing
            self._update_movement_progress(landmarks)
            
            # Check completion with improved accuracy
            if self._check_movement_completion():
                logger.info(f"Completed calibration state: {self.current_state}")
                self._advance_state()
                
            # Get updated instruction
            instruction = self.get_current_instruction()
            instruction['animation'] = self._get_animation_hints()
            return instruction
            
        except Exception as e:
            logger.error(f"Error updating calibration: {str(e)}")
            return self.get_current_instruction()
        
    def _update_movement_progress(self, landmarks):
        """Update movement detection with improved accuracy"""
        if self.current_state == CalibrationState.HEAD_TURN:
            nose = landmarks[0]
            left_ear = landmarks[7]
            right_ear = landmarks[8]
            
            # Calculate head rotation with enhanced detection
            head_rotation = abs(nose[0] - (left_ear[0] + right_ear[0])/2)
            smoothed = self._smooth_movement(
                head_rotation / self.movement_thresholds['head_rotation'],
                'rotation'
            )
            
            self.movement_progress['left'] = min(1.0, smoothed)
            self.movement_progress['right'] = self.movement_progress['left']
            
        elif self.current_state == CalibrationState.ARMS_RAISE:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Enhanced arm raise detection
            left_raise = max(0, left_shoulder[1] - left_wrist[1])
            right_raise = max(0, right_shoulder[1] - right_wrist[1])
            
            smoothed = self._smooth_movement(
                max(left_raise, right_raise) / self.movement_thresholds['arm_raise'],
                'raise'
            )
            
            self.movement_progress['up'] = min(1.0, smoothed)
            self.movement_progress['down'] = self.movement_progress['up']
            
        elif self.current_state == CalibrationState.BODY_TURN:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            hip_center = (landmarks[23] + landmarks[24]) / 2
            
            # Improved rotation detection
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            hip_alignment = abs(hip_center[0] - (left_shoulder[0] + right_shoulder[0])/2)
            
            rotation = min(1.0, (1 - shoulder_width) / self.movement_thresholds['body_rotation'] + 
                         hip_alignment / self.movement_thresholds['body_rotation'])
            
            smoothed = self._smooth_movement(rotation, 'rotation')
            
            self.movement_progress['left'] = min(1.0, smoothed)
            self.movement_progress['right'] = self.movement_progress['left']
            
        elif self.current_state == CalibrationState.SQUAT:
            hip_center = (landmarks[23] + landmarks[24]) / 2
            knee_center = (landmarks[25] + landmarks[26]) / 2
            ankle_center = (landmarks[27] + landmarks[28]) / 2
            
            # Enhanced squat detection
            hip_height = hip_center[1]
            knee_height = knee_center[1]
            ankle_height = ankle_center[1]
            
            squat_depth = (knee_height - ankle_height) / (hip_height - ankle_height)
            smoothed = self._smooth_movement(
                squat_depth / self.movement_thresholds['squat_depth'],
                'squat'
            )
            
            self.movement_progress['down'] = min(1.0, smoothed)
            self.movement_progress['up'] = self.movement_progress['down']
            
    def _check_movement_completion(self):
        """Check movement completion with improved thresholds"""
        completion_threshold = 0.85  # Slightly reduced threshold for better user experience
        
        if self.current_state == CalibrationState.HEAD_TURN:
            return (self.movement_progress['left'] >= completion_threshold and 
                   self.movement_progress['right'] >= completion_threshold)
        elif self.current_state == CalibrationState.ARMS_RAISE:
            return (self.movement_progress['up'] >= completion_threshold and 
                   self.movement_progress['down'] >= completion_threshold)
        elif self.current_state == CalibrationState.BODY_TURN:
            return (self.movement_progress['left'] >= completion_threshold and 
                   self.movement_progress['right'] >= completion_threshold)
        elif self.current_state == CalibrationState.SQUAT:
            return (self.movement_progress['down'] >= completion_threshold and 
                   self.movement_progress['up'] >= completion_threshold)
        return False
        
    def _advance_state(self):
        """Advance to next calibration state with transition logging"""
        states = list(CalibrationState)
        current_index = states.index(self.current_state)
        
        if current_index < len(states) - 1:
            prev_state = self.current_state
            self.current_state = states[current_index + 1]
            logger.info(f"Advancing from {prev_state} to {self.current_state}")
            self.state_start_time = time.time()
            self.reset_movement_progress()
        else:
            logger.info("Calibration completed successfully")
            self.current_state = CalibrationState.COMPLETED
            self.state_start_time = None
            
    def _get_animation_hints(self):
        """Provide enhanced animation hints for smoother visualization"""
        return {
            CalibrationState.HEAD_TURN: {
                'type': 'oscillate',
                'axis': 'x',
                'range': [-25, 25],
                'duration': 3000,
                'easing': 'easeInOutQuad'
            },
            CalibrationState.ARMS_RAISE: {
                'type': 'linear',
                'axis': 'y',
                'range': [0, 160],
                'duration': 4000,
                'easing': 'easeInOutCubic'
            },
            CalibrationState.BODY_TURN: {
                'type': 'oscillate',
                'axis': 'y',
                'range': [-35, 35],
                'duration': 3000,
                'easing': 'easeInOutQuad'
            },
            CalibrationState.SQUAT: {
                'type': 'ease',
                'axis': 'y',
                'range': [0, -25],
                'duration': 2000,
                'easing': 'easeInOutQuad'
            }
        }.get(self.current_state, None)
        
    def get_current_instruction(self):
        """Get current instruction with enhanced progress tracking"""
        instruction = self.instructions[self.current_state]
        
        # Calculate detailed progress
        progress = 0
        if self.current_state != CalibrationState.NOT_STARTED and self.current_state != CalibrationState.COMPLETED:
            if self.current_state in [CalibrationState.HEAD_TURN, CalibrationState.BODY_TURN]:
                progress = int((self.movement_progress['left'] + self.movement_progress['right']) * 50)
            elif self.current_state == CalibrationState.ARMS_RAISE:
                progress = int((self.movement_progress['up'] + self.movement_progress['down']) * 50)
            elif self.current_state == CalibrationState.SQUAT:
                progress = int((self.movement_progress['down'] + self.movement_progress['up']) * 50)
                
        # Add time remaining if applicable
        time_remaining = None
        if self.state_start_time and self.current_state in self.state_durations:
            elapsed = time.time() - self.state_start_time
            remaining = max(0, self.state_durations[self.current_state] - elapsed)
            time_remaining = int(remaining)
            
        return {
            "state": self.current_state.value,
            "title": instruction["title"],
            "text": instruction["text"],
            "details": instruction["details"],
            "success_criteria": instruction["success_criteria"],
            "progress": progress,
            "time_remaining": time_remaining,
            "movement_progress": self.movement_progress
        }
