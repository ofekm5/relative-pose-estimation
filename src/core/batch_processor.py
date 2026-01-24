"""
Batch processing of multiple image pairs for trajectory estimation.
High-level component for processing sequences of frames.
"""

from pathlib import Path
import numpy as np
from ..utils.image_loader import load_image_pair
from ..utils.geometry import (
    rotation_to_euler, euler_to_rotation, CONVENTION_YUP
)


class BatchProcessor:
    """
    Processes sequences of image pairs to estimate camera trajectory.

    Coordinates image loading, pose estimation, and trajectory accumulation
    across multiple consecutive frames.
    """

    def __init__(self, images_dir, pose_estimator, ground_truth_loader,
                 euler_convention=CONVENTION_YUP):
        """
        Initialize batch processor.

        Args:
            images_dir: Directory containing image files
            pose_estimator: PoseEstimator instance for computing relative poses
            ground_truth_loader: GroundTruthLoader instance for accessing frame data
            euler_convention: Euler angle convention ('yup' or 'zyx')
        """
        self.images_dir = Path(images_dir)
        self.pose_estimator = pose_estimator
        self.gt_loader = ground_truth_loader
        self.euler_convention = euler_convention

    def process_sequence(self, frame_indices):
        """
        Process a sequence of consecutive frame pairs.

        Args:
            frame_indices: Array of frame indices to process (must be sorted)

        Returns:
            dict: {
                'frames': List of frame indices (excluding first frame),
                'roll': Array of estimated roll angles (degrees),
                'pitch': Array of estimated pitch angles (degrees),
                'yaw': Array of estimated yaw angles (degrees),
                'R': List of rotation matrices in world frame,
                't': List of translation vectors (direction only)
            }

        Raises:
            RuntimeError: If pose estimation fails for any frame pair
        """
        if len(frame_indices) < 2:
            raise ValueError("Need at least 2 frames to process")

        results = {
            'frames': [],
            'roll': [],
            'pitch': [],
            'yaw': [],
            'R': [],
            't': []
        }

        # Process consecutive pairs
        for i in range(len(frame_indices) - 1):
            frame1_idx = frame_indices[i]
            frame2_idx = frame_indices[i + 1]

            # Load image pair
            img1_path = self.images_dir / f"{frame1_idx:06d}.png"
            img2_path = self.images_dir / f"{frame2_idx:06d}.png"

            img1, img2 = load_image_pair(str(img1_path), str(img2_path), to_gray=True)

            # Get ground truth orientation of first frame
            gt_pose1 = self.gt_loader.get_pose(frame1_idx)
            prev_roll = gt_pose1['roll']
            prev_pitch = gt_pose1['pitch']
            prev_yaw = gt_pose1['yaw']

            # Convert ground truth orientation to rotation matrix
            R_prev_world = euler_to_rotation(prev_yaw, prev_pitch, prev_roll,
                                             convention=self.euler_convention)

            # Estimate relative pose (with VP refinement if enabled)
            R_rel, t_rel = self.pose_estimator.estimate(img1, img2, R_prev=R_prev_world)

            # Compose to get world rotation of second frame
            # R_rel represents rotation from camera1 to camera2
            # R_cam2 = R_cam1 @ R_rel
            R_new_world = R_prev_world @ R_rel

            # Convert back to Euler angles
            yaw_est, pitch_est, roll_est = rotation_to_euler(R_new_world,
                                                              convention=self.euler_convention)

            # Store results
            results['frames'].append(frame2_idx)
            results['roll'].append(roll_est)
            results['pitch'].append(pitch_est)
            results['yaw'].append(yaw_est)
            results['R'].append(R_new_world)
            results['t'].append(t_rel)

        # Convert lists to arrays
        results['roll'] = np.array(results['roll'])
        results['pitch'] = np.array(results['pitch'])
        results['yaw'] = np.array(results['yaw'])

        return results

    def process_at_interval(self, step=15):
        """
        Process frames at regular intervals from ground truth data.

        Args:
            step: Frame interval (e.g., step=15 processes frames 0, 15, 30, ...)

        Returns:
            dict: Same format as process_sequence()
        """
        frame_indices = self.gt_loader.get_frame_indices(step=step)
        return self.process_sequence(frame_indices)

    def get_image_path(self, frame_idx):
        """
        Get path to image file for a given frame index.

        Args:
            frame_idx: Frame index

        Returns:
            Path: Path to image file
        """
        return self.images_dir / f"{frame_idx:06d}.png"
