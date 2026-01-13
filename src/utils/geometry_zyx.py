"""
Geometry utilities for ZYX camera convention.
For phone camera data compatibility.
"""

import numpy as np
import math


def rotation_to_euler_zyx_camera(R):
    """
    Convert rotation matrix to Euler angles with ZYX camera convention.

    This matches the ArUco ground truth generation (qr_pose_frames.py).
    Rotation order: R = Rz(yaw) * Ry(pitch) * Rx(roll)

    Args:
        R: Rotation matrix (3x3 numpy array)

    Returns:
        tuple: (yaw_deg, pitch_deg, roll_deg) in degrees
    """
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0

    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def euler_to_rotation_zyx_camera(yaw_deg, pitch_deg, roll_deg):
    """
    Convert Euler angles to rotation matrix with ZYX camera convention.

    Rotation order: R = Rz(yaw) * Ry(pitch) * Rx(roll)

    Args:
        yaw_deg: Yaw angle in degrees
        pitch_deg: Pitch angle in degrees
        roll_deg: Roll angle in degrees

    Returns:
        np.ndarray: Rotation matrix (3x3)
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # ZYX order: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,             cp*cr           ]
    ])

    return R
