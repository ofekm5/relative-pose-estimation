"""
Camera calibration and intrinsic parameters management.
High-level component for computing camera matrix K.
"""

import numpy as np
from pathlib import Path

class CameraCalibration:
    """
    Manages camera intrinsic parameters and computes calibration matrix.

    The camera matrix K is scaled from base intrinsics according to actual image size.
    Base intrinsics are from the simulator's camera model.
    """

    def __init__(self,
                 camera_matrix=None,
                 calibration_file=None,
                 fx_base=924.82939686,
                 fy_base=920.4766382,
                 cx_base=468.24930789,
                 cy_base=353.65863024,
                 base_width=960,
                 base_height=720):
        """
        Initialize camera calibration.

        Args:
            camera_matrix: Direct 3x3 camera matrix K (overrides other params)
            calibration_file: Path to .npz file with 'K' key (overrides base params)
            fx_base: Base focal length in x direction (pixels)
            fy_base: Base focal length in y direction (pixels)
            cx_base: Base principal point x coordinate (pixels)
            cy_base: Base principal point y coordinate (pixels)
            base_width: Base image width (pixels)
            base_height: Base image height (pixels)
        """
        self.fixed_K = None

        # Priority 1: Direct camera matrix
        if camera_matrix is not None:
            self.fixed_K = np.array(camera_matrix, dtype=np.float64)
            if self.fixed_K.shape != (3, 3):
                raise ValueError(f"camera_matrix must be 3x3, got {self.fixed_K.shape}")

        # Priority 2: Load from calibration file
        elif calibration_file is not None:
            cal_path = Path(calibration_file)
            if not cal_path.exists():
                raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
            data = np.load(cal_path)
            if 'K' not in data:
                raise KeyError(f"Calibration file must contain 'K' key, found: {list(data.keys())}")
            self.fixed_K = data['K']

        # Priority 3: Use base parameters for scaling
        self.fx_base = fx_base
        self.fy_base = fy_base
        self.cx_base = cx_base
        self.cy_base = cy_base
        self.base_width = base_width
        self.base_height = base_height

    def get_matrix(self, image_width=None, image_height=None):
        """Get camera intrinsic matrix K."""
        if self.fixed_K is not None:
            return self.fixed_K

        if image_width is None or image_height is None:
            raise ValueError("image_width and image_height required when using base parameters")

        scale_x = image_width / self.base_width
        scale_y = image_height / self.base_height

        fx = self.fx_base * scale_x
        fy = self.fy_base * scale_y
        cx = self.cx_base * scale_x
        cy = self.cy_base * scale_y

        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)

        return K

    def get_matrix_from_image(self, image):
        """
        Compute camera matrix from an image (uses image shape).

        Args:
            image: Image as numpy array (shape: (height, width) or (height, width, channels))

        Returns:
            np.ndarray: Camera matrix K (3x3)
        """
        if len(image.shape) == 2:
            height, width = image.shape
        else:
            height, width = image.shape[:2]

        return self.get_matrix(width, height)
