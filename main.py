"""
Entry point for running the pose estimation pipeline.
"""

from src.pipeline import PoseEstimationPipeline

def main():
    # Initialize pipeline
    # pipeline = PoseEstimationPipeline(
    #     data_dir="simulator-data",
    #     results_dir="results",
    #     feature_method="ORB",
    #     norm_type="Hamming",
    #     max_matches=500
    # )

    pipeline = PoseEstimationPipeline(
        data_dir="phone-data",
        calibration_file="phone-data/calibration_scaled.npz",
        results_dir="results",
        feature_method="ORB",
        norm_type="Hamming",
        max_matches=500
    )

    # # Custom intrinsics directly
    # K = np.array([[1322.4, 0, 1005.8], [0, 1328.5, 616.4], [0, 0, 1]])
    # pipeline = PoseEstimationPipeline(
    #     images_dir="phone_camera/forward_with_stuff/images",
    #     data_dir="phone_camera/forward_with_stuff",
    #     camera_matrix=K
    # )

    # Setup components
    pipeline.setup()

    # Run full pipeline
    results = pipeline.run(
        step=5,
        create_plot=True,
        create_video=True,
        video_fps=5
    )

    print("\n[INFO] Pipeline results keys:", results.keys())

    # Optionally test a single pair
    # pipeline.run_single_pair(0, 15, show_debug=True)


if __name__ == "__main__":
    main()
