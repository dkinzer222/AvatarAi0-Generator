import numpy as np
import cv2

class SMPLXRenderer:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.prev_landmarks = None
        self.interpolation_frames = 5
        self.scale = 200  # Scale factor for 3D to 2D projection
        
    def render_avatar(self, landmarks):
        """Render a simplified 3D skeleton avatar using MediaPipe landmarks"""
        if landmarks is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Create a blank image
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Extract position and visibility data
        positions = landmarks[:, :3]  # x, y, z
        visibility = landmarks[:, 3]  # visibility values
        
        # Perform interpolation if we have previous landmarks
        if self.prev_landmarks is not None:
            prev_positions = self.prev_landmarks[:, :3]
            interpolated_frames = self._interpolate_poses(prev_positions, positions)
            image = self._render_3d_skeleton(interpolated_frames[-1], visibility)
        else:
            image = self._render_3d_skeleton(positions, visibility)
        
        # Store current landmarks for next frame
        self.prev_landmarks = landmarks.copy()
        
        return image
            
    def _render_3d_skeleton(self, landmarks, visibility):
        """Render an enhanced 3D skeleton with depth visualization"""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Define enhanced connections for better visualization
        connections = [
            # Torso
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
        
        # Project 3D points to 2D with perspective
        points_2d = self._project_3d_to_2d(landmarks)
        
        # Draw connections with depth-based coloring
        for start_idx, end_idx in connections:
            if start_idx < len(points_2d) and end_idx < len(points_2d):
                if visibility[start_idx] > 0.5 and visibility[end_idx] > 0.5:
                    start = tuple(points_2d[start_idx].astype(int))
                    end = tuple(points_2d[end_idx].astype(int))
                    
                    # Calculate depth for color gradient
                    z_avg = (landmarks[start_idx][2] + landmarks[end_idx][2]) / 2
                    color_intensity = int(255 * (1 + z_avg))
                    color = (0, min(255, color_intensity), min(255, color_intensity + 50))
                    
                    cv2.line(image, start, end, color, 2, cv2.LINE_AA)
        
        # Draw joints with depth-based size
        for i, (point, vis) in enumerate(zip(points_2d, visibility)):
            if vis > 0.5:
                point = tuple(point.astype(int))
                z_depth = landmarks[i][2]
                radius = int(5 * (1 + z_depth))
                color_intensity = int(255 * vis)
                cv2.circle(image, point, radius, (color_intensity, 0, 0), -1, cv2.LINE_AA)
        
        return image
    
    def _project_3d_to_2d(self, points_3d):
        """Project 3D points to 2D space with perspective effect"""
        points_2d = points_3d[:, :2] * self.scale
        
        # Add perspective effect based on Z coordinate
        z_scale = 1 + np.tanh(points_3d[:, 2:3])
        points_2d = points_2d * z_scale
        
        # Center the projection
        points_2d = points_2d + np.array([self.width/2, self.height/2])
        return points_2d
    
    def _interpolate_poses(self, prev_landmarks, curr_landmarks):
        """Create smooth transitions between poses"""
        frames = []
        for i in range(self.interpolation_frames):
            t = (i + 1) / (self.interpolation_frames + 1)
            # Use smooth interpolation curve
            t = 0.5 * (1 - np.cos(t * np.pi))
            interpolated = prev_landmarks + t * (curr_landmarks - prev_landmarks)
            frames.append(interpolated)
        return frames
