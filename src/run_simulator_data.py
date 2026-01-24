"""
Simulator data pose estimation pipeline.

Usage:
    python -m src.run_simulator_data [--step STEP] [--no-plot] [--no-video]
"""

import argparse
from .pipeline import PoseEstimationPipeline


def main():
    parser = argparse.ArgumentParser(description="Simulator Data Pose Estimation Pipeline")
    parser.add_argument('--step', type=int, default=15, help='Frame step interval (default: 15)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    parser.add_argument('--no-video', action='store_true', help='Skip video generation')
    parser.add_argument('--video-fps', type=int, default=10, help='Video FPS (default: 10)')

    args = parser.parse_args()

    # Initialize pipeline with simulator data configuration
    pipeline = PoseEstimationPipeline(
        data_dir="evaluation-runs/simulator-data/data",
        gt_filename="camera_poses.txt",
        calibration_file=None,  # Use default CameraCalibration
        results_dir="evaluation-runs/simulator-data/results",
        feature_method="ORB",
        norm_type="Hamming",
        max_matches=500,
        euler_convention="yup"  # Simulator uses YUP convention
    )

    pipeline.setup()

    results = pipeline.run(
        step=args.step,
        create_plot=not args.no_plot,
        create_video=not args.no_video,
        video_fps=args.video_fps
    )

    print("\n[INFO] Pipeline completed. Results saved to evaluation-runs/simulator-data/results/")
    return results


if __name__ == "__main__":
    main()
