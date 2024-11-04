import numpy as np
import cv2

class AvatarRenderer:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.prev_landmarks = None
        self.interpolation_frames = 5
        
    def render_avatar(self, landmarks):
        """Render a simple but enhanced skeleton avatar"""
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
            for frame in interpolated_frames:
                image = self._render_enhanced_skeleton(frame, visibility, image.copy())
        else:
            image = self._render_enhanced_skeleton(positions, visibility, image)
        
        # Store current landmarks for next frame
        self.prev_landmarks = landmarks.copy()
        
        return image
            
    def _render_enhanced_skeleton(self, landmarks, visibility, image):
        """Render an enhanced skeleton with more connections and better visualization"""
        # Define enhanced connections for better visualization
        connections = [
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),
            (15, 17), (15, 19), (15, 21),  # Left hand
            (16, 18), (16, 20), (16, 22),  # Right hand
            # Legs
            (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
            # Face connections
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
        ]
        
        # Scale landmarks to image dimensions
        scaled_landmarks = landmarks.copy()
        scaled_landmarks[:, 0] *= self.width
        scaled_landmarks[:, 1] *= self.height
        
        # Draw connections with gradient colors
        for start_idx, end_idx in connections:
            if start_idx < len(scaled_landmarks) and end_idx < len(scaled_landmarks):
                # Check visibility of both points
                if visibility[start_idx] > 0.5 and visibility[end_idx] > 0.5:
                    start_point = tuple(scaled_landmarks[start_idx][:2].astype(int))
                    end_point = tuple(scaled_landmarks[end_idx][:2].astype(int))
                    
                    # Create gradient color based on depth (z-coordinate)
                    z_avg = (landmarks[start_idx][2] + landmarks[end_idx][2]) / 2
                    color_intensity = int(255 * (1 + z_avg))
                    color = (0, color_intensity, color_intensity)
                    
                    cv2.line(image, start_point, end_point, color, 2)
        
        # Draw joints with different sizes based on depth and visibility
        for i, (landmark, vis) in enumerate(zip(scaled_landmarks, visibility)):
            if vis > 0.5:  # Only draw visible landmarks
                point = tuple(landmark[:2].astype(int))
                depth = landmark[2]
                radius = int(4 * (1 + depth))
                color_intensity = int(255 * vis)
                cv2.circle(image, point, radius, (color_intensity, 0, 0), -1)
            
        return image
    
    def _interpolate_poses(self, prev_landmarks, curr_landmarks):
        """Create smooth transitions between poses"""
        frames = []
        for i in range(self.interpolation_frames):
            t = (i + 1) / (self.interpolation_frames + 1)
            interpolated = prev_landmarks + t * (curr_landmarks - prev_landmarks)
            frames.append(interpolated)
        return frames
