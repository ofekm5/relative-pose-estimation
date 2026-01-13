"""
Geometry utilities for rotation and coordinate transformations.
Low-level helper functions for pose estimation.
"""

import numpy as np


def rotation_to_euler_yup(R):
    """
    Convert rotation matrix to Euler angles (yaw, pitch, roll) with Y-up convention.

    Rotation order: R = Ry(yaw) * Rx(pitch) * Rz(roll)

    Args:
        R: Rotation matrix (3x3 numpy array)

    Returns:
        tuple: (yaw_deg, pitch_deg, roll_deg) in degrees
    """
    # Extract pitch from R[2,1] = sin(pitch)
    pitch = np.arcsin(R[2, 1])

    # Handle gimbal lock (pitch = ±90°)
    if abs(R[2, 1]) > 0.9999:
        # Gimbal lock case
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    else:
        # Normal case
        # Yaw from first column (cos(pitch) removes coupling)
        yaw = np.arctan2(-R[2, 0], R[0, 0])
        # Roll from row/column after removing yaw/pitch influence
        roll = np.arctan2(R[1, 0], R[1, 1])

    # Convert to degrees
    yaw_deg = np.rad2deg(yaw)
    pitch_deg = np.rad2deg(pitch)
    roll_deg = np.rad2deg(roll)

    return yaw_deg, pitch_deg, roll_deg


def euler_to_rotation_yup(yaw_deg, pitch_deg, roll_deg):
    """
    Convert Euler angles to rotation matrix with Y-up convention.

    Rotation order: R = Ry(yaw) * Rx(pitch) * Rz(roll)

    Args:
        yaw_deg: Yaw angle in degrees
        pitch_deg: Pitch angle in degrees
        roll_deg: Roll angle in degrees

    Returns:
        np.ndarray: Rotation matrix (3x3)
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    # Rotation around Y axis (yaw)
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Rotation around X axis (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    # Rotation around Z axis (roll)
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

    # Combined rotation
    R = Ry @ Rx @ Rz

    return R


def rotation_error(R_est, R_gt):
    """
    Compute rotation error between estimated and ground truth rotation matrices.

    Args:
        R_est: Estimated rotation matrix (3x3)
        R_gt: Ground truth rotation matrix (3x3)

    Returns:
        float: Rotation error in degrees
    """
    R_diff = R_est @ R_gt.T
    trace = np.trace(R_diff)

    # Clamp to avoid numerical issues with arccos
    cos_angle = (trace - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.rad2deg(angle_rad)

    return angle_deg


def translation_direction_error(t_est, t_gt):
    """
    Compute translation direction error (ignoring scale).

    Args:
        t_est: Estimated translation vector (3,)
        t_gt: Ground truth translation vector (3,)

    Returns:
        float: Direction error in degrees
    """
    # Normalize vectors (scale is ambiguous in monocular)
    t_est_norm = t_est.flatten() / np.linalg.norm(t_est)
    t_gt_norm = t_gt.flatten() / np.linalg.norm(t_gt)

    # Compute angle between directions
    dot_product = np.dot(t_est_norm, t_gt_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle_rad = np.arccos(dot_product)
    angle_deg = np.rad2deg(angle_rad)

    return angle_deg
