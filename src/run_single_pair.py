"""
Single image pair pose estimation.

Usage:
    python -m src.run_single_pair [--img1 PATH] [--img2 PATH] [--calibration CALIB_FILE]
"""

import argparse
import numpy as np
import cv2

from .core.camera_calibration import CameraCalibration
from .core.pose_estimator import PoseEstimator


def main():
    parser = argparse.ArgumentParser(description="Single Pair Pose Estimation")
    parser.add_argument('--img1', default='evaluation-runs/single-pair/images/000000.png',
                        help='Path to first image (default: evaluation-runs/single-pair/images/000000.png)')
    parser.add_argument('--img2', default='evaluation-runs/single-pair/images/000015.png',
                        help='Path to second image (default: evaluation-runs/single-pair/images/000015.png)')
    parser.add_argument('--calibration', '-c', help='Optional path to calibration .npz file (must contain "K" matrix)')

    args = parser.parse_args()

    # Get image paths from arguments
    img1_path = args.img1
    img2_path = args.img2

    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        raise FileNotFoundError(f"Could not load image: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Could not load image: {img2_path}")

    # Get camera matrix
    if args.calibration:
        calib_data = np.load(args.calibration)
        K = calib_data['K']
    else:
        # Use default calibration scaled to image size
        calibration = CameraCalibration()
        K = calibration.get_matrix(img1.shape[1], img1.shape[0])

    # Create pose estimator
    estimator = PoseEstimator(
        camera_matrix=K,
        feature_method="ORB",
        nfeatures=4000,
        use_vp_refinement=True
    )

    # Estimate pose
    R, t = estimator.estimate(img1, img2)

    # Display results
    print("\n=== Relative Pose Estimation ===")
    print(f"\nImages: {img1_path} -> {img2_path}")
    print(f"\nRotation Matrix R:")
    print(R)
    print(f"\nTranslation t (direction only):")
    print(t.flatten())
    return R, t



if __name__ == "__main__":
    main()
