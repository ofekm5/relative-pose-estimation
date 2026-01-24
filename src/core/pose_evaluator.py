"""
Pose evaluation and error metrics computation.
High-level component for comparing estimated poses against ground truth.
"""

import numpy as np
import pandas as pd
from ..utils.geometry import (
    rotation_error, translation_direction_error, euler_to_rotation, CONVENTION_YUP
)


class PoseEvaluator:
    """
    Evaluates estimated camera poses against ground truth data.

    Computes error metrics for rotation and translation, and provides
    summary statistics for trajectory accuracy assessment.
    """

    def __init__(self, ground_truth_loader, euler_convention=CONVENTION_YUP):
        """
        Initialize pose evaluator.

        Args:
            ground_truth_loader: GroundTruthLoader instance for accessing GT poses
            euler_convention: Euler angle convention ('yup' or 'zyx')
        """
        self.gt_loader = ground_truth_loader
        self.euler_convention = euler_convention

    def evaluate_sequence(self, estimated_results):
        """
        Evaluate estimated poses against ground truth.

        Args:
            estimated_results: Dict from BatchProcessor with keys:
                'frames', 'roll', 'pitch', 'yaw', 'R', 't'

        Returns:
            dict: {
                'frames': Frame indices,
                'roll_error': Roll angle errors (degrees),
                'pitch_error': Pitch angle errors (degrees),
                'yaw_error': Yaw angle errors (degrees),
                'rotation_error': Overall rotation errors (degrees),
                'translation_dir_error': Translation direction errors (degrees),
                'gt_roll': Ground truth roll angles,
                'gt_pitch': Ground truth pitch angles,
                'gt_yaw': Ground truth yaw angles,
                'est_roll': Estimated roll angles,
                'est_pitch': Estimated pitch angles,
                'est_yaw': Estimated yaw angles
            }
        """
        frames = estimated_results['frames']
        est_roll = estimated_results['roll']
        est_pitch = estimated_results['pitch']
        est_yaw = estimated_results['yaw']
        est_R = estimated_results['R']
        est_t = estimated_results['t']

        # Initialize result arrays
        roll_errors = []
        pitch_errors = []
        yaw_errors = []
        rotation_errors = []
        translation_dir_errors = []

        gt_roll_vals = []
        gt_pitch_vals = []
        gt_yaw_vals = []

        # Evaluate each frame
        prev_gt_pos = None
        for i, frame_idx in enumerate(frames):
            # Get ground truth pose
            gt_pose = self.gt_loader.get_pose(frame_idx)

            gt_roll = gt_pose['roll']
            gt_pitch = gt_pose['pitch']
            gt_yaw = gt_pose['yaw']
            gt_pos = np.array([gt_pose['x'], gt_pose['y'], gt_pose['z']])

            # Compute angle errors
            roll_error_val = abs(est_roll[i] - gt_roll)
            pitch_error_val = abs(est_pitch[i] - gt_pitch)
            yaw_error_val = abs(est_yaw[i] - gt_yaw)

            # Wrap angle errors to [-180, 180]
            roll_error_val = self._wrap_angle_error(roll_error_val)
            pitch_error_val = self._wrap_angle_error(pitch_error_val)
            yaw_error_val = self._wrap_angle_error(yaw_error_val)

            # Compute rotation matrix error
            R_gt = euler_to_rotation(gt_yaw, gt_pitch, gt_roll,
                                     convention=self.euler_convention)
            rot_error = rotation_error(est_R[i], R_gt)

            # Store results
            roll_errors.append(roll_error_val)
            pitch_errors.append(pitch_error_val)
            yaw_errors.append(yaw_error_val)
            rotation_errors.append(rot_error)

            gt_roll_vals.append(gt_roll)
            gt_pitch_vals.append(gt_pitch)
            gt_yaw_vals.append(gt_yaw)

            # Compute translation direction error using GT delta position
            if prev_gt_pos is not None:
                gt_delta = gt_pos - prev_gt_pos
                trans_err = translation_direction_error(est_t[i], gt_delta)
                translation_dir_errors.append(trans_err)
            else:
                # First frame has no previous position
                translation_dir_errors.append(0.0)

            prev_gt_pos = gt_pos

        return {
            'frames': frames,
            'roll_error': np.array(roll_errors),
            'pitch_error': np.array(pitch_errors),
            'yaw_error': np.array(yaw_errors),
            'rotation_error': np.array(rotation_errors),
            'translation_dir_error': np.array(translation_dir_errors),
            'gt_roll': np.array(gt_roll_vals),
            'gt_pitch': np.array(gt_pitch_vals),
            'gt_yaw': np.array(gt_yaw_vals),
            'est_roll': est_roll,
            'est_pitch': est_pitch,
            'est_yaw': est_yaw
        }

    def compute_summary_statistics(self, evaluation_results):
        """
        Compute summary statistics for evaluation results.

        Args:
            evaluation_results: Dict from evaluate_sequence()

        Returns:
            dict: Summary statistics including mean, std, median, max errors
        """
        stats = {}

        for metric in ['roll_error', 'pitch_error', 'yaw_error', 'rotation_error', 'translation_dir_error']:
            errors = evaluation_results[metric]

            stats[f'{metric}_mean'] = np.mean(errors)
            stats[f'{metric}_std'] = np.std(errors)
            stats[f'{metric}_median'] = np.median(errors)
            stats[f'{metric}_max'] = np.max(errors)
            stats[f'{metric}_min'] = np.min(errors)

        return stats

    def create_comparison_dataframe(self, evaluation_results):
        """
        Create a pandas DataFrame with side-by-side comparison.

        Args:
            evaluation_results: Dict from evaluate_sequence()

        Returns:
            pd.DataFrame: Comparison data with columns for GT, EST, and errors
        """
        df = pd.DataFrame({
            'frame': evaluation_results['frames'],
            'gt_roll': evaluation_results['gt_roll'],
            'gt_pitch': evaluation_results['gt_pitch'],
            'gt_yaw': evaluation_results['gt_yaw'],
            'est_roll': evaluation_results['est_roll'],
            'est_pitch': evaluation_results['est_pitch'],
            'est_yaw': evaluation_results['est_yaw'],
            'roll_error': evaluation_results['roll_error'],
            'pitch_error': evaluation_results['pitch_error'],
            'yaw_error': evaluation_results['yaw_error'],
            'rotation_error': evaluation_results['rotation_error'],
            'translation_dir_error': evaluation_results['translation_dir_error']
        })

        return df

    @staticmethod
    def _wrap_angle_error(error_deg):
        """
        Wrap angle error to [-180, 180] range.

        Args:
            error_deg: Angle error in degrees

        Returns:
            float: Wrapped error in degrees
        """
        wrapped = ((error_deg + 180) % 360) - 180
        return abs(wrapped)

    def print_summary(self, evaluation_results):
        """
        Print evaluation summary to console.

        Args:
            evaluation_results: Dict from evaluate_sequence()
        """
        stats = self.compute_summary_statistics(evaluation_results)

        print("\n" + "="*60)
        print("POSE ESTIMATION EVALUATION SUMMARY")
        print("="*60)

        print(f"\nNumber of frames evaluated: {len(evaluation_results['frames'])}")

        print("\nRotation Errors (degrees):")
        print(f"  Mean:   {stats['rotation_error_mean']:.2f}")
        print(f"  Std:    {stats['rotation_error_std']:.2f}")
        print(f"  Median: {stats['rotation_error_median']:.2f}")
        print(f"  Max:    {stats['rotation_error_max']:.2f}")
        print(f"  Min:    {stats['rotation_error_min']:.2f}")

        print("\nRoll Errors (degrees):")
        print(f"  Mean:   {stats['roll_error_mean']:.2f}")
        print(f"  Std:    {stats['roll_error_std']:.2f}")

        print("\nPitch Errors (degrees):")
        print(f"  Mean:   {stats['pitch_error_mean']:.2f}")
        print(f"  Std:    {stats['pitch_error_std']:.2f}")

        print("\nYaw Errors (degrees):")
        print(f"  Mean:   {stats['yaw_error_mean']:.2f}")
        print(f"  Std:    {stats['yaw_error_std']:.2f}")

        print("\nTranslation Direction Errors (degrees):")
        print(f"  Mean:   {stats['translation_dir_error_mean']:.2f}")
        print(f"  Std:    {stats['translation_dir_error_std']:.2f}")

        print("\n" + "="*60 + "\n")
