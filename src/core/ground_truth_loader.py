"""
Ground truth data loading and management.
High-level component for loading and accessing ground truth camera poses.
"""

from pathlib import Path
import pandas as pd
import numpy as np


class GroundTruthLoader:
    """
    Loads and provides access to ground truth camera poses.

    Ground truth file format (camera_poses.txt):
        frame  x  y  z  roll  pitch  yaw
        0      ...
        15     ...
        ...
    """

    def __init__(self, gt_path):
        """
        Initialize ground truth loader.

        Args:
            gt_path: Path to ground truth file (camera_poses.txt)
        """
        self.gt_path = Path(gt_path)
        self.df = None

    def load(self):
        """
        Load ground truth data from file.

        Returns:
            pd.DataFrame: Ground truth data with columns:
                frame, x, y, z, roll, pitch, yaw
        """
        self.df = pd.read_csv(self.gt_path, sep=r'\s+')
        return self.df

    def get_pose(self, frame_idx):
        """
        Get ground truth pose for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            dict: Pose data with keys: x, y, z, roll, pitch, yaw, frame
        """
        if self.df is None:
            raise RuntimeError("Ground truth not loaded. Call load() first.")

        row = self.df[self.df["frame"] == frame_idx].iloc[0]

        return {
            'frame': int(row['frame']),
            'x': float(row['x']),
            'y': float(row['y']),
            'z': float(row['z']),
            'roll': float(row['roll']),
            'pitch': float(row['pitch']),
            'yaw': float(row['yaw'])
        }

    def get_frame_indices(self, step=1):
        """
        Get frame indices at regular intervals.

        Args:
            step: Frame interval (e.g., step=15 returns every 15th frame)

        Returns:
            np.ndarray: Array of frame indices
        """
        if self.df is None:
            raise RuntimeError("Ground truth not loaded. Call load() first.")

        frames = self.df[self.df["frame"] % step == 0]["frame"].values
        return frames

    def get_all_frames(self):
        """
        Get all frame indices.

        Returns:
            np.ndarray: Array of all frame indices
        """
        if self.df is None:
            raise RuntimeError("Ground truth not loaded. Call load() first.")

        return self.df["frame"].values

    def get_trajectory(self, step=1):
        """
        Get trajectory (positions only) at regular intervals.

        Args:
            step: Frame interval

        Returns:
            np.ndarray: Array of positions (N, 3) with columns [x, y, z]
        """
        if self.df is None:
            raise RuntimeError("Ground truth not loaded. Call load() first.")

        df_sub = self.df[self.df["frame"] % step == 0]
        positions = df_sub[["x", "y", "z"]].values

        return positions

    def get_orientations(self, step=1):
        """
        Get orientations (roll, pitch, yaw) at regular intervals.

        Args:
            step: Frame interval

        Returns:
            np.ndarray: Array of orientations (N, 3) with columns [roll, pitch, yaw]
        """
        if self.df is None:
            raise RuntimeError("Ground truth not loaded. Call load() first.")

        df_sub = self.df[self.df["frame"] % step == 0]
        orientations = df_sub[["roll", "pitch", "yaw"]].values

        return orientations
