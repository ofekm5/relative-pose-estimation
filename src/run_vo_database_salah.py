"""
Phone data pose estimation pipeline.

Usage:
    python -m src.run_phone_data [--step STEP] [--no-plot] [--no-video]
"""

import argparse
from .pipeline import PoseEstimationPipeline
import numpy as np

def load_K(calib: dict) -> np.ndarray:
    if "K" in calib:
        K = calib["K"]
    elif "intrinsic_matrix" in calib:
        K = calib["intrinsic_matrix"]
    else:
        raise KeyError(f"Calibration file must contain 'K' or 'intrinsic_matrix'. Found: {list(calib.keys())}")

    K = np.asarray(K, dtype=float)
    # Handle both 3x3 and flattened 9-element cases
    if K.shape == (9,):
        K = K.reshape(3, 3)
    if K.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must be 3x3. Got shape {K.shape}")
    return K


def main():
    parser = argparse.ArgumentParser(description="Phone Data Pose Estimation Pipeline")
    parser.add_argument('--step', type=int, default=5, help='Frame step interval (default: 5)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    parser.add_argument('--no-video', action='store_true', help='Skip video generation')
    parser.add_argument('--video-fps', type=int, default=5, help='Video FPS (default: 5)')

    args = parser.parse_args()

    K = load_K(np.load("evaluation-runs/vo_dataset_salah/data/calibration.npz"))
    # Initialize pipeline with phone data configuration
    pipeline = PoseEstimationPipeline(
        data_dir="evaluation-runs/vo_dataset_salah/data",
        gt_filename="camera_poses_zyx.txt",
        camera_matrix =K,
        results_dir="evaluation-runs/vo_dataset_salah/results",
        feature_method="ORB",
        norm_type="Hamming",
        max_matches=500,
        euler_convention="zyx"  # Phone uses ZYX convention
    )

    pipeline.setup()

    results = pipeline.run(
        step=args.step,
        create_plot=not args.no_plot,
        create_video=not args.no_video,
        video_fps=args.video_fps
    )

    print("\n[INFO] Pipeline completed. Results saved to evaluation-runs/phone-data/results/")
    return results


if __name__ == "__main__":
    main()
