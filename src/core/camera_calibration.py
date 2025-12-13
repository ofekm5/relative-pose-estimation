"""
Camera calibration and intrinsic parameters management.
High-level component for computing camera matrix K.
"""

import numpy as np


class CameraCalibration:
    """
    Manages camera intrinsic parameters and computes calibration matrix.

    The camera matrix K is scaled from base intrinsics according to actual image size.
    Base intrinsics are from the simulator's camera model.
    """

    def __init__(self,
                 fx_base=924.82939686,
                 fy_base=920.4766382,
                 cx_base=468.24930789,
                 cy_base=353.65863024,
                 base_width=960,
                 base_height=720):
        """
        Initialize camera calibration with base parameters.

        Args:
            fx_base: Base focal length in x direction (pixels)
            fy_base: Base focal length in y direction (pixels)
            cx_base: Base principal point x coordinate (pixels)
            cy_base: Base principal point y coordinate (pixels)
            base_width: Base image width (pixels)
            base_height: Base image height (pixels)
        """
        self.fx_base = fx_base
        self.fy_base = fy_base
        self.cx_base = cx_base
        self.cy_base = cy_base
        self.base_width = base_width
        self.base_height = base_height

    def get_matrix(self, image_width, image_height):
        """
        Compute camera intrinsic matrix K scaled to actual image size.

        Args:
            image_width: Actual image width in pixels
            image_height: Actual image height in pixels

        Returns:
            np.ndarray: Camera matrix K (3x3)
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        """
        # Compute scaling factors
        scale_x = image_width / self.base_width
        scale_y = image_height / self.base_height

        # Scale intrinsic parameters
        fx = self.fx_base * scale_x
        fy = self.fy_base * scale_y
        cx = self.cx_base * scale_x
        cy = self.cy_base * scale_y

        # Build camera matrix
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
