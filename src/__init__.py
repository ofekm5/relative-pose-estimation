"""
Source code for 6-DoF Relative Pose Estimation system.
"""

# Main pipeline
from .pipeline import PoseEstimationPipeline

# Core components
from .core.camera_calibration import CameraCalibration
from .core.ground_truth_loader import GroundTruthLoader
from .core.pose_estimator import PoseEstimator
from .core.batch_processor import BatchProcessor
from .core.pose_evaluator import PoseEvaluator
from .core.visualizer import Visualizer

# Utilities
from .utils.image_loader import load_image, load_image_pair
from .utils.geometry import (
    rotation_to_euler_yup,
    euler_to_rotation_yup,
    rotation_error,
    translation_direction_error
)

__all__ = [
    # Pipeline
    'PoseEstimationPipeline',
    # Core
    'CameraCalibration',
    'GroundTruthLoader',
    'PoseEstimator',
    'BatchProcessor',
    'PoseEvaluator',
    'Visualizer',
    # Utils
    'load_image',
    'load_image_pair',
    'rotation_to_euler_yup',
    'euler_to_rotation_yup',
    'rotation_error',
    'translation_direction_error',
]
