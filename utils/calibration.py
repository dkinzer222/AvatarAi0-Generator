import time
from enum import Enum

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
            CalibrationState.HEAD_TURN: 5,
            CalibrationState.ARMS_RAISE: 5,
            CalibrationState.BODY_TURN: 5,
            CalibrationState.SQUAT: 5
        }
        self.instructions = {
            CalibrationState.NOT_STARTED: {
                "title": "Start Calibration",
                "text": "Stand in a clear space, facing the camera",
                "success_criteria": "Click Start when ready"
            },
            CalibrationState.HEAD_TURN: {
                "title": "Head Movement",
                "text": "Slowly turn your head left to right",
                "success_criteria": "Complete head rotation"
            },
            CalibrationState.ARMS_RAISE: {
                "title": "Arm Movement",
                "text": "Raise both arms to shoulder height and back down",
                "success_criteria": "Complete arm raise"
            },
            CalibrationState.BODY_TURN: {
                "title": "Body Rotation",
                "text": "Slowly turn your body 45° left and right",
                "success_criteria": "Complete body rotation"
            },
            CalibrationState.SQUAT: {
                "title": "Squat Movement",
                "text": "Perform a partial squat and return to standing",
                "success_criteria": "Complete squat motion"
            },
            CalibrationState.COMPLETED: {
                "title": "Calibration Complete",
                "text": "Great job! Avatar is now calibrated",
                "success_criteria": "All movements completed"
            }
        }
        
    def start_calibration(self):
        """Start the calibration process"""
        self.current_state = CalibrationState.HEAD_TURN
        self.state_start_time = time.time()
        return self.get_current_instruction()
        
    def update_calibration(self, landmarks):
        """Update calibration state based on pose landmarks"""
        if self.current_state == CalibrationState.COMPLETED:
            return self.get_current_instruction()
            
        if self.state_start_time is None:
            return self.get_current_instruction()
            
        # Check if current state duration has elapsed
        elapsed_time = time.time() - self.state_start_time
        if elapsed_time >= self.state_durations[self.current_state]:
            self._advance_state()
            
        return self.get_current_instruction()
        
    def _advance_state(self):
        """Move to the next calibration state"""
        states = list(CalibrationState)
        current_index = states.index(self.current_state)
        
        if current_index < len(states) - 1:
            self.current_state = states[current_index + 1]
            self.state_start_time = time.time()
        else:
            self.current_state = CalibrationState.COMPLETED
            self.state_start_time = None
            
    def get_current_instruction(self):
        """Get the current calibration instruction"""
        instruction = self.instructions[self.current_state]
        progress = 0
        
        if self.state_start_time is not None:
            elapsed_time = time.time() - self.state_start_time
            total_time = self.state_durations.get(self.current_state, 0)
            progress = min(100, int((elapsed_time / total_time) * 100)) if total_time > 0 else 0
            
        return {
            "state": self.current_state.value,
            "title": instruction["title"],
            "text": instruction["text"],
            "success_criteria": instruction["success_criteria"],
            "progress": progress
        }
