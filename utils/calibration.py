import time
from enum import Enum
import numpy as np

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
        self.movement_thresholds = {
            'head_rotation': 0.3,
            'arm_raise': 0.4,
            'body_rotation': 0.25,
            'squat_depth': 0.2
        }
        self.movement_progress = {
            'left': 0,
            'right': 0,
            'up': 0,
            'down': 0
        }
        self.instructions = {
            CalibrationState.NOT_STARTED: {
                "title": "🎯 Start Calibration",
                "text": "Let's set up your avatar for the best experience! Find a well-lit space with room to move.",
                "details": [
                    "Stand 2-3 meters from camera",
                    "Ensure your full body is visible",
                    "Wear contrasting clothing",
                    "Clear the area around you"
                ],
                "success_criteria": "Ready to begin? Click Start!"
            },
            CalibrationState.HEAD_TURN: {
                "title": "👤 Head Movement Calibration",
                "text": "Let's start with natural head movements",
                "details": [
                    "Start by looking straight ahead",
                    "Slowly turn head left (3s)",
                    "Return to center (2s)",
                    "Slowly turn head right (3s)",
                    "Keep shoulders still and relaxed"
                ],
                "success_criteria": "Smooth head rotation detected"
            },
            CalibrationState.ARMS_RAISE: {
                "title": "💪 Arm Range Calibration",
                "text": "Now let's check your arm movement range",
                "details": [
                    "Start with arms relaxed at sides",
                    "Slowly raise both arms forward",
                    "Continue until arms are overhead",
                    "Hold briefly at the top",
                    "Lower arms gradually back down"
                ],
                "success_criteria": "Full arm motion range captured"
            },
            CalibrationState.BODY_TURN: {
                "title": "🔄 Body Rotation Check",
                "text": "Time to check your torso mobility",
                "details": [
                    "Keep feet planted shoulder-width apart",
                    "Start facing forward",
                    "Rotate upper body left slowly",
                    "Return to center, pause briefly",
                    "Rotate upper body right slowly"
                ],
                "success_criteria": "Smooth torso rotation detected"
            },
            CalibrationState.SQUAT: {
                "title": "⬇️ Lower Body Calibration",
                "text": "Final step - checking lower body movement",
                "details": [
                    "Stand with feet shoulder-width apart",
                    "Keep your back straight",
                    "Slowly bend knees (partial squat)",
                    "Hold position briefly",
                    "Return to standing position smoothly"
                ],
                "success_criteria": "Controlled squat motion detected"
            },
            CalibrationState.COMPLETED: {
                "title": "✨ Calibration Complete!",
                "text": "Perfect! Your avatar is now fully calibrated and ready to mirror your movements.",
                "details": [
                    "All movement ranges captured successfully",
                    "Avatar tracking optimized for your body",
                    "Movement detection fine-tuned",
                    "You can recalibrate anytime if needed"
                ],
                "success_criteria": "All movements successfully recorded"
            }
        }
        
    def start_calibration(self):
        """Start the calibration process"""
        self.current_state = CalibrationState.HEAD_TURN
        self.state_start_time = time.time()
        self.reset_movement_progress()
        return self.get_current_instruction()
        
    def reset_movement_progress(self):
        """Reset movement progress tracking"""
        self.movement_progress = {
            'left': 0,
            'right': 0,
            'up': 0,
            'down': 0
        }
        
    def update_calibration(self, landmarks):
        """Update calibration state based on pose landmarks"""
        if landmarks is None or self.current_state == CalibrationState.COMPLETED:
            return self.get_current_instruction()
            
        if self.state_start_time is None:
            return self.get_current_instruction()
            
        # Update movement progress based on current state
        self._update_movement_progress(landmarks)
        
        # Check if current movement is complete
        if self._check_movement_completion():
            self._advance_state()
            
        # Get current instruction with updated progress
        instruction = self.get_current_instruction()
        
        # Add smooth animation and transition hints
        if self.current_state != CalibrationState.NOT_STARTED and self.current_state != CalibrationState.COMPLETED:
            instruction['animation'] = self._get_animation_hints()
            
        return instruction
        
    def _update_movement_progress(self, landmarks):
        """Update movement progress with improved accuracy"""
        if self.current_state == CalibrationState.HEAD_TURN:
            # Enhanced head rotation detection using nose and ear positions
            nose = landmarks[0]
            left_ear = landmarks[7]
            right_ear = landmarks[8]
            
            # Calculate normalized head rotation
            head_rotation = abs(nose[0] - (left_ear[0] + right_ear[0])/2)
            # Apply smoothing
            self.movement_progress['left'] = min(1.0, max(self.movement_progress['left'],
                                                        head_rotation / self.movement_thresholds['head_rotation']))
            self.movement_progress['right'] = self.movement_progress['left']
            
        elif self.current_state == CalibrationState.ARMS_RAISE:
            # Improved arm raise detection with shoulder reference
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Calculate vertical movement with shoulder offset
            left_raise = max(0, left_shoulder[1] - left_wrist[1])
            right_raise = max(0, right_shoulder[1] - right_wrist[1])
            
            # Update progress with momentum
            self.movement_progress['up'] = min(1.0, max(self.movement_progress['up'],
                                                      max(left_raise, right_raise) / self.movement_thresholds['arm_raise']))
            self.movement_progress['down'] = self.movement_progress['up']
            
        elif self.current_state == CalibrationState.BODY_TURN:
            # Enhanced body rotation detection
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            hip_center = (landmarks[23] + landmarks[24]) / 2
            
            # Calculate rotation using shoulder-hip relationship
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            hip_alignment = abs(hip_center[0] - (left_shoulder[0] + right_shoulder[0])/2)
            
            rotation = min(1.0, (1 - shoulder_width) / self.movement_thresholds['body_rotation'] + 
                         hip_alignment / self.movement_thresholds['body_rotation'])
            
            self.movement_progress['left'] = min(1.0, max(self.movement_progress['left'], rotation))
            self.movement_progress['right'] = self.movement_progress['left']
            
        elif self.current_state == CalibrationState.SQUAT:
            # Improved squat detection with knee tracking
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            # Calculate squat depth with ankle reference
            hip_height = (left_hip[1] + right_hip[1]) / 2
            knee_height = (left_knee[1] + right_knee[1]) / 2
            ankle_height = (left_ankle[1] + right_ankle[1]) / 2
            
            # Normalized squat depth
            squat_depth = abs((hip_height - knee_height) / (hip_height - ankle_height))
            
            self.movement_progress['down'] = min(1.0, max(self.movement_progress['down'],
                                                        squat_depth / self.movement_thresholds['squat_depth']))
            self.movement_progress['up'] = self.movement_progress['down']
            
    def _check_movement_completion(self):
        """Check if current movement requirements are met"""
        if self.current_state == CalibrationState.HEAD_TURN:
            return self.movement_progress['left'] >= 0.9 and self.movement_progress['right'] >= 0.9
        elif self.current_state == CalibrationState.ARMS_RAISE:
            return self.movement_progress['up'] >= 0.9 and self.movement_progress['down'] >= 0.9
        elif self.current_state == CalibrationState.BODY_TURN:
            return self.movement_progress['left'] >= 0.9 and self.movement_progress['right'] >= 0.9
        elif self.current_state == CalibrationState.SQUAT:
            return self.movement_progress['down'] >= 0.9 and self.movement_progress['up'] >= 0.9
        return False
        
    def _advance_state(self):
        """Move to the next calibration state"""
        states = list(CalibrationState)
        current_index = states.index(self.current_state)
        
        if current_index < len(states) - 1:
            self.current_state = states[current_index + 1]
            self.state_start_time = time.time()
            self.reset_movement_progress()
        else:
            self.current_state = CalibrationState.COMPLETED
            self.state_start_time = None
            
    def _get_animation_hints(self):
        """Provide animation hints for smooth movement visualization"""
        return {
            CalibrationState.HEAD_TURN: {
                'type': 'oscillate',
                'axis': 'x',
                'range': [-30, 30],
                'duration': 3000
            },
            CalibrationState.ARMS_RAISE: {
                'type': 'linear',
                'axis': 'y',
                'range': [0, 180],
                'duration': 4000
            },
            CalibrationState.BODY_TURN: {
                'type': 'oscillate',
                'axis': 'y',
                'range': [-45, 45],
                'duration': 3000
            },
            CalibrationState.SQUAT: {
                'type': 'ease',
                'axis': 'y',
                'range': [0, -30],
                'duration': 2000
            }
        }.get(self.current_state, None)
        
    def get_current_instruction(self):
        """Get the current calibration instruction with detailed progress"""
        instruction = self.instructions[self.current_state]
        
        # Calculate overall progress
        progress = 0
        if self.current_state != CalibrationState.NOT_STARTED and self.current_state != CalibrationState.COMPLETED:
            if self.current_state in [CalibrationState.HEAD_TURN, CalibrationState.BODY_TURN]:
                progress = int((self.movement_progress['left'] + self.movement_progress['right']) * 50)
            elif self.current_state == CalibrationState.ARMS_RAISE:
                progress = int((self.movement_progress['up'] + self.movement_progress['down']) * 50)
            elif self.current_state == CalibrationState.SQUAT:
                progress = int((self.movement_progress['down'] + self.movement_progress['up']) * 50)
            
        return {
            "state": self.current_state.value,
            "title": instruction["title"],
            "text": instruction["text"],
            "details": instruction["details"],
            "success_criteria": instruction["success_criteria"],
            "progress": progress,
            "movement_progress": self.movement_progress
        }
