"""
Main pipeline for 6-DoF relative pose estimation.
Orchestrates all components: calibration, pose estimation, evaluation, and visualization.
"""

from pathlib import Path
from .core.camera_calibration import CameraCalibration
from .core.ground_truth_loader import GroundTruthLoader
from .core.pose_estimator import PoseEstimator
from .core.batch_processor import BatchProcessor
from .core.pose_evaluator import PoseEvaluator
from .core.visualizer import Visualizer
from .utils.image_loader import load_image


class PoseEstimationPipeline:
    """
    High-level pipeline for camera pose estimation from image sequences.

    Coordinates all components from camera calibration to final visualization.
    """

    def __init__(self,
                 data_dir="data",
                 results_dir="results",
                 gt_filename="camera_poses.txt",
                 feature_method="ORB",
                 norm_type="Hamming",
                 max_matches=500):
        """
        Initialize pose estimation pipeline.

        Args:
            data_dir: Directory containing images/ and ground truth file (default: "data")
            results_dir: Directory for output files (plots, videos, CSVs) (default: "results")
            gt_filename: Filename of ground truth poses file
            feature_method: Feature detector method ('ORB' or 'SIFT')
            norm_type: Distance norm for matching ('Hamming' for ORB, 'L2' for SIFT)
            max_matches: Maximum number of matches to use for pose estimation
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.gt_path = self.data_dir / gt_filename
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Store configuration
        self.feature_method = feature_method
        self.norm_type = norm_type
        self.max_matches = max_matches

        # Components (initialized in setup)
        self.camera_calibration = None
        self.gt_loader = None
        self.pose_estimator = None
        self.batch_processor = None
        self.pose_evaluator = None
        self.visualizer = None

    def setup(self):
        """
        Initialize all pipeline components.

        Must be called before running the pipeline.
        """
        # 1. Setup ground truth loader
        self.gt_loader = GroundTruthLoader(self.gt_path)
        self.gt_loader.load()

        # 2. Setup camera calibration
        self.camera_calibration = CameraCalibration()

        # Get camera matrix from a sample image
        sample_frames = self.gt_loader.get_all_frames()
        sample_frame_idx = sample_frames[0]
        sample_img_path = self.images_dir / f"{sample_frame_idx:06d}.png"
        sample_img = load_image(str(sample_img_path), to_gray=True)
        K = self.camera_calibration.get_matrix_from_image(sample_img)

        # 3. Setup pose estimator
        self.pose_estimator = PoseEstimator(
            camera_matrix=K,
            feature_method=self.feature_method,
            norm_type=self.norm_type,
            max_matches=self.max_matches
        )

        # 4. Setup batch processor
        self.batch_processor = BatchProcessor(
            images_dir=self.images_dir,
            pose_estimator=self.pose_estimator,
            ground_truth_loader=self.gt_loader
        )

        # 5. Setup evaluator
        self.pose_evaluator = PoseEvaluator(
            ground_truth_loader=self.gt_loader
        )

        # 6. Setup visualizer
        self.visualizer = Visualizer(
            output_dir=self.results_dir
        )

        print(f"[INFO] Pipeline initialized")
        print(f"[INFO] Data directory: {self.data_dir}")
        print(f"[INFO] Images directory: {self.images_dir}")
        print(f"[INFO] Ground truth: {self.gt_path}")
        print(f"[INFO] Results directory: {self.results_dir}")
        print(f"[INFO] Feature method: {self.feature_method}")
        print(f"[INFO] Camera matrix K computed from image size: {sample_img.shape}")

    def run(self, step=15, create_plot=True, create_video=False, video_fps=10):
        """
        Run the complete pose estimation pipeline.

        Args:
            step: Frame interval to process (e.g., step=15 processes frames 0, 15, 30, ...)
            create_plot: Whether to create 3D trajectory plot
            create_video: Whether to create annotated video
            video_fps: Frames per second for output video

        Returns:
            dict: Pipeline results including estimated poses and evaluation metrics
        """
        if self.batch_processor is None:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        print(f"\n[INFO] Running pipeline with step={step}")

        # 1. Process image sequence
        print(f"[INFO] Processing image sequence...")
        estimated_results = self.batch_processor.process_at_interval(step=step)
        print(f"[INFO] Processed {len(estimated_results['frames'])} frame pairs")

        # 2. Evaluate against ground truth
        print(f"[INFO] Evaluating against ground truth...")
        evaluation_results = self.pose_evaluator.evaluate_sequence(estimated_results)

        # 3. Print evaluation summary
        self.pose_evaluator.print_summary(evaluation_results)

        # 4. Create comparison dataframe
        comparison_df = self.pose_evaluator.create_comparison_dataframe(evaluation_results)
        csv_path = self.results_dir / "evaluation_results.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"[INFO] Evaluation results saved to: {csv_path}")

        # 5. Create visualizations
        if create_plot:
            print(f"[INFO] Creating 3D trajectory plot...")
            gt_trajectory_full = self.gt_loader.get_trajectory(step=1)  # All frames for path
            gt_trajectory_filtered = self.gt_loader.get_trajectory(step=step)  # Step-filtered for arrows
            gt_orientations_filtered = self.gt_loader.get_orientations(step=step)  # GT orientations for arrows
            self.visualizer.plot_3d_trajectory(
                gt_trajectory_full=gt_trajectory_full,
                gt_trajectory_filtered=gt_trajectory_filtered,
                gt_orientations_filtered=gt_orientations_filtered,
                evaluation_results=evaluation_results,
                step=step
            )

        if create_video:
            print(f"[INFO] Creating annotated video...")
            self.visualizer.create_video(
                images_dir=self.images_dir,
                evaluation_results=evaluation_results,
                output_filename="pose_comparison.mp4",
                fps=video_fps
            )

        print(f"\n[INFO] Pipeline complete!")

        return {
            'estimated': estimated_results,
            'evaluation': evaluation_results,
            'comparison_df': comparison_df
        }

    def run_single_pair(self, frame1_idx, frame2_idx, show_debug=False):
        """
        Run pose estimation on a single image pair.

        Args:
            frame1_idx: First frame index
            frame2_idx: Second frame index
            show_debug: Whether to print debug information

        Returns:
            dict: Pose estimation results for the pair
        """
        if self.pose_estimator is None:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        print(f"\n[INFO] Processing single pair: frames {frame1_idx} -> {frame2_idx}")

        # Load images
        from .utils.image_loader import load_image_pair
        img1_path = self.images_dir / f"{frame1_idx:06d}.png"
        img2_path = self.images_dir / f"{frame2_idx:06d}.png"
        img1, img2 = load_image_pair(str(img1_path), str(img2_path), to_gray=True)

        # Estimate pose
        if show_debug:
            result = self.pose_estimator.estimate_with_debug(img1, img2)
            R, t = result['R'], result['t']
            print(f"[DEBUG] Number of matches: {result['num_matches']}")
            print(f"[DEBUG] Inliers: {result['inliers']}")
        else:
            R, t = self.pose_estimator.estimate(img1, img2)

        # Convert to Euler angles
        from .utils.geometry import rotation_to_euler_yup
        yaw, pitch, roll = rotation_to_euler_yup(R)

        print(f"[INFO] Estimated relative pose:")
        print(f"  Yaw:   {yaw:.2f}°")
        print(f"  Pitch: {pitch:.2f}°")
        print(f"  Roll:  {roll:.2f}°")

        # Get ground truth for comparison
        gt_pose1 = self.gt_loader.get_pose(frame1_idx)
        gt_pose2 = self.gt_loader.get_pose(frame2_idx)

        print(f"\n[INFO] Ground truth poses:")
        print(f"  Frame {frame1_idx}: yaw={gt_pose1['yaw']:.2f}°, pitch={gt_pose1['pitch']:.2f}°, roll={gt_pose1['roll']:.2f}°")
        print(f"  Frame {frame2_idx}: yaw={gt_pose2['yaw']:.2f}°, pitch={gt_pose2['pitch']:.2f}°, roll={gt_pose2['roll']:.2f}°")

        return {
            'R': R,
            't': t,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'gt_pose1': gt_pose1,
            'gt_pose2': gt_pose2
        }


def main():
    """
    Example usage of the pipeline.
    """
    # Initialize pipeline
    pipeline = PoseEstimationPipeline(
        data_dir="data",
        results_dir="results",
        feature_method="ORB",
        norm_type="Hamming",
        max_matches=500
    )

    # Setup components
    pipeline.setup()

    # Run full pipeline
    results = pipeline.run(
        step=15,
        create_plot=True,
        create_video=True,
        video_fps=5
    )

    print("\n[INFO] Pipeline results keys:", results.keys())

    # Optionally test a single pair
    # pipeline.run_single_pair(0, 15, show_debug=True)


if __name__ == "__main__":
    main()
