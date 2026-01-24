"""
Entry point for pose estimation - supports single pair and pipeline modes.

Usage:
    python main.py single img1.png img2.png [--calibration calib.npz]
    python main.py pipeline --data-dir phone-data --step 5
"""

import argparse
import numpy as np
import cv2

from src.core.camera_calibration import CameraCalibration
from src.core.pose_estimator import PoseEstimator
from src.utils.geometry import rotation_to_euler_yup


def run_single_pair(img1_path: str, img2_path: str, calibration_file: str = None) -> dict:
    """
    Estimate relative pose between two images.

    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        calibration_file: Optional path to calibration .npz file with camera matrix K

    Returns:
        dict with 'R', 't', 'yaw', 'pitch', 'roll'
    """
    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        raise FileNotFoundError(f"Could not load image: {img1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Could not load image: {img2_path}")

    # Get camera matrix
    if calibration_file:
        calib_data = np.load(calibration_file)
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

    # Convert to Euler angles
    yaw, pitch, roll = rotation_to_euler_yup(R)

    return {
        'R': R,
        't': t,
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll
    }


def run_pipeline(data_dir: str, gt_filename: str, calibration_file: str,
                 step: int, create_plot: bool, create_video: bool, video_fps: int,
                 euler_convention: str = "zyx"):
    """Run full pipeline on image sequence."""
    from src.pipeline import PoseEstimationPipeline

    pipeline = PoseEstimationPipeline(
        data_dir=data_dir,
        gt_filename=gt_filename,
        calibration_file=calibration_file,
        results_dir="results",
        feature_method="ORB",
        norm_type="Hamming",
        max_matches=500,
        euler_convention=euler_convention
    )

    pipeline.setup()

    results = pipeline.run(
        step=step,
        create_plot=create_plot,
        create_video=create_video,
        video_fps=video_fps
    )

    print("\n[INFO] Pipeline results keys:", results.keys())
    return results


def main():
    parser = argparse.ArgumentParser(
        description="6-DoF Relative Pose Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Single pair subcommand
    single_parser = subparsers.add_parser('single', help='Estimate pose from two images')
    single_parser.add_argument('img1', help='Path to first image')
    single_parser.add_argument('img2', help='Path to second image')
    single_parser.add_argument('--calibration', '-c', help='Path to calibration .npz file')

    # Pipeline subcommand
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline on image sequence')
    pipeline_parser.add_argument('--data-dir', default='phone-data', help='Data directory')
    pipeline_parser.add_argument('--gt-filename', default='camera_poses_zyx.txt', help='Ground truth filename')
    pipeline_parser.add_argument('--calibration', default='phone-data/calibration_scaled.npz',
                                 help='Calibration file path')
    pipeline_parser.add_argument('--step', type=int, default=5, help='Frame step interval')
    pipeline_parser.add_argument('--convention', choices=['yup', 'zyx'], default='zyx',
                                 help='Euler angle convention (yup=simulator, zyx=phone)')
    pipeline_parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    pipeline_parser.add_argument('--no-video', action='store_true', help='Skip video generation')
    pipeline_parser.add_argument('--video-fps', type=int, default=5, help='Video FPS')

    args = parser.parse_args()

    if args.mode == 'single':
        result = run_single_pair(args.img1, args.img2, args.calibration)

        print("\n=== Relative Pose Estimation ===")
        print(f"\nRotation Matrix R:")
        print(result['R'])
        print(f"\nTranslation t (direction only):")
        print(result['t'].flatten())
        print(f"\nEuler Angles (degrees):")
        print(f"  Yaw:   {result['yaw']:.2f}")
        print(f"  Pitch: {result['pitch']:.2f}")
        print(f"  Roll:  {result['roll']:.2f}")

    elif args.mode == 'pipeline':
        run_pipeline(
            data_dir=args.data_dir,
            gt_filename=args.gt_filename,
            calibration_file=args.calibration,
            step=args.step,
            create_plot=not args.no_plot,
            create_video=not args.no_video,
            video_fps=args.video_fps,
            euler_convention=args.convention
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
