import numpy as np
import cv2
import time

class SMPLXRenderer:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.prev_landmarks = None
        self.interpolation_frames = 5
        self.scale = 200
        self.last_render_time = 0
        self.min_render_interval = 1/30  # Cap at 30 FPS
        
        # Add customization parameters with defaults
        self.avatar_color = (200, 200, 200)  # Default color (RGB)
        self.avatar_size = 1.0  # Scale factor
        self.line_thickness = 2
        self.joint_size = 1.0
        self.style = "solid"  # solid, dashed, or gradient
        self.expression_colors = {
            'neutral': (200, 200, 200),
            'closed_eyes': (255, 165, 0),
            'open_mouth': (0, 255, 255),
            'raised_eyebrows': (255, 0, 255),
            'frown': (128, 0, 128)
        }
        # Initialize frame buffer for motion smoothing
        self.frame_buffer = []
        self.max_buffer_size = 5
        
    def set_customization(self, color=None, size=None, style=None, line_thickness=None, joint_size=None):
        """Update avatar customization parameters with validation"""
        if color is not None:
            if isinstance(color, tuple) and len(color) == 3:
                self.avatar_color = tuple(map(lambda x: max(0, min(255, x)), color))
        if size is not None:
            self.avatar_size = max(0.5, min(2.0, float(size)))
        if style is not None and style in ["solid", "dashed", "gradient"]:
            self.style = style
        if line_thickness is not None:
            self.line_thickness = max(1, min(5, int(line_thickness)))
        if joint_size is not None:
            self.joint_size = max(0.5, min(2.0, float(joint_size)))
            
    def _smooth_motion(self, landmarks):
        """Apply motion smoothing using exponential moving average"""
        if landmarks is None:
            return None
            
        # Add current frame to buffer
        self.frame_buffer.append(landmarks)
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
            
        # Apply exponential weights for smoother motion
        weights = np.exp(np.linspace(-1, 0, len(self.frame_buffer)))
        weights /= weights.sum()
        
        # Calculate weighted average of positions
        smoothed = np.zeros_like(self.frame_buffer[0])
        for i, frame in enumerate(self.frame_buffer):
            smoothed += frame * weights[i]
            
        return smoothed
        
    def render_avatar(self, landmarks, expression=None):
        """Render a simplified 3D skeleton avatar using MediaPipe landmarks with optimizations"""
        current_time = time.time()
        if current_time - self.last_render_time < self.min_render_interval:
            return None  # Skip frame if too soon
            
        if landmarks is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        # Apply motion smoothing
        smoothed_landmarks = self._smooth_motion(landmarks)
        if smoothed_landmarks is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        # Create frame buffer
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Extract position and visibility data
        positions = smoothed_landmarks[:, :3]
        visibility = smoothed_landmarks[:, 3]
        
        # Apply size scaling
        positions = positions * self.avatar_size
        
        # Perform interpolation if we have previous landmarks
        if self.prev_landmarks is not None:
            prev_positions = self.prev_landmarks[:, :3] * self.avatar_size
            interpolated_frames = self._interpolate_poses(prev_positions, positions)
            image = self._render_3d_skeleton(interpolated_frames[-1], visibility, expression)
        else:
            image = self._render_3d_skeleton(positions, visibility, expression)
            
        # Store current landmarks for next frame
        self.prev_landmarks = smoothed_landmarks.copy()
        self.last_render_time = current_time
        
        return image
            
    def _render_3d_skeleton(self, landmarks, visibility, expression=None):
        """Render an enhanced 3D skeleton with optimized drawing"""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Define optimized connections for better visualization
        connections = [
            # Torso (core structure)
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),
            # Hands
            (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
            # Legs
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32),
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8)
        ]
        
        # Project 3D points to 2D with enhanced perspective
        points_2d = self._project_3d_to_2d(landmarks)
        
        # Pre-calculate shared values
        line_color = self.avatar_color
        line_thickness = self.line_thickness
        
        # Draw connections efficiently
        for start_idx, end_idx in connections:
            if start_idx < len(points_2d) and end_idx < len(points_2d):
                if visibility[start_idx] > 0.5 and visibility[end_idx] > 0.5:
                    start = tuple(points_2d[start_idx].astype(int))
                    end = tuple(points_2d[end_idx].astype(int))
                    
                    # Skip if points are outside frame
                    if not (0 <= start[0] < self.width and 0 <= start[1] < self.height and
                           0 <= end[0] < self.width and 0 <= end[1] < self.height):
                        continue
                    
                    # Calculate color based on depth
                    if self.style == "gradient":
                        z_avg = (landmarks[start_idx][2] + landmarks[end_idx][2]) / 2
                        color = tuple(min(255, int(c * (1 + z_avg))) for c in self.avatar_color)
                    else:
                        color = line_color
                    
                    # Draw optimized lines
                    if self.style == "dashed":
                        self._draw_dashed_line(image, start, end, color, line_thickness)
                    else:
                        cv2.line(image, start, end, color, line_thickness, cv2.LINE_AA)
        
        # Draw joints efficiently
        for i, (point, vis) in enumerate(zip(points_2d, visibility)):
            if vis > 0.5:
                point = tuple(map(int, point))
                if not (0 <= point[0] < self.width and 0 <= point[1] < self.height):
                    continue
                    
                z_depth = landmarks[i][2]
                radius = max(1, min(20, int(5 * self.joint_size * (1 + z_depth))))
                
                color = self.expression_colors.get(expression, self.avatar_color) if i < 11 else self.avatar_color
                cv2.circle(image, point, radius, color, -1, cv2.LINE_AA)
        
        return image
    
    def _draw_dashed_line(self, image, start, end, color, thickness):
        """Draw an optimized dashed line between two points"""
        dash_length = 10
        gap_length = 5
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist == 0:
            return
            
        dx = dx/dist
        dy = dy/dist
        
        curr_x, curr_y = start
        step = dash_length + gap_length
        
        while dist > 0:
            next_x = int(curr_x + dx * min(dash_length, dist))
            next_y = int(curr_y + dy * min(dash_length, dist))
            
            # Skip if points are outside frame
            if (0 <= int(curr_x) < self.width and 0 <= int(curr_y) < self.height and
                0 <= next_x < self.width and 0 <= next_y < self.height):
                cv2.line(image, (int(curr_x), int(curr_y)), (next_x, next_y), 
                        color, thickness, cv2.LINE_AA)
            
            curr_x = curr_x + dx * step
            curr_y = curr_y + dy * step
            dist -= step
    
    def _project_3d_to_2d(self, points_3d):
        """Project 3D points to 2D space with enhanced perspective effect"""
        points_2d = points_3d[:, :2] * self.scale
        
        # Apply improved perspective effect based on Z coordinate
        z_scale = 1 + np.tanh(points_3d[:, 2:3])
        points_2d = points_2d * z_scale
        
        # Center the projection with bounds checking
        points_2d = points_2d + np.array([self.width/2, self.height/2])
        return points_2d
    
    def _interpolate_poses(self, prev_landmarks, curr_landmarks):
        """Create smooth transitions between poses with easing function"""
        frames = []
        for i in range(self.interpolation_frames):
            t = (i + 1) / (self.interpolation_frames + 1)
            # Use smooth easing function
            t = 0.5 * (1 - np.cos(t * np.pi))
            interpolated = prev_landmarks + t * (curr_landmarks - prev_landmarks)
            frames.append(interpolated)
        return frames
