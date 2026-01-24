"""
Visualization utilities for trajectory and pose data.
High-level component for creating plots and videos.
"""

from pathlib import Path
import numpy as np
import cv2
import plotly.graph_objects as go
from ..utils.geometry import euler_to_rotation_yup


class Visualizer:
    """
    Creates visualizations for camera trajectories and pose estimates.

    Supports 3D interactive plots and video generation with overlay annotations.
    """

    def __init__(self, output_dir):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save output files (plots, videos)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_3d_trajectory(self, gt_trajectory_full, gt_trajectory_filtered,
                          gt_orientations_filtered, evaluation_results,
                          arrow_scale=0.3, step=15):
        """
        Create 3D interactive plot of ground truth and estimated trajectories.

        Args:
            gt_trajectory_full: Full ground truth positions (all frames) for path line
            gt_trajectory_filtered: Filtered GT positions (at step interval) for arrows
            gt_orientations_filtered: Filtered GT orientations (roll, pitch, yaw) for arrows
            evaluation_results: Dict from PoseEvaluator with EST angles
            arrow_scale: Scale factor for orientation arrows
            step: Frame step interval for labeling

        Returns:
            str: Path to saved HTML file
        """
        # Extract data
        gt_positions_full = gt_trajectory_full  # All frames for path
        gt_positions_filtered = gt_trajectory_filtered  # Step-filtered for arrows
        est_roll = evaluation_results['est_roll']
        est_pitch = evaluation_results['est_pitch']
        est_yaw = evaluation_results['est_yaw']

        # Build orientation vectors
        dirs_gt = []
        dirs_est = []
        labels = []

        # Build GT directions from actual filtered ground truth orientations
        # gt_orientations_filtered has shape (N, 3) with columns [roll, pitch, yaw]
        for i in range(len(gt_orientations_filtered)):
            roll, pitch, yaw = gt_orientations_filtered[i]
            d_gt = self._rpy_to_direction(roll, pitch, yaw)
            dirs_gt.append(d_gt * arrow_scale)

            frame_idx = i * step
            labels.append(f"{frame_idx}-{frame_idx + step}")

        dirs_gt = np.array(dirs_gt)

        # Build EST directions for all estimates
        for i in range(len(est_roll)):
            d_est = self._rpy_to_direction(est_roll[i], est_pitch[i], est_yaw[i])
            dirs_est.append(d_est * arrow_scale)

        # Prepend first GT arrow to EST (EST starts from second frame)
        dirs_est = [dirs_gt[0]] + dirs_est
        dirs_est = np.array(dirs_est)

        # Create plot
        fig = go.Figure()

        COLOR_GT = "red"
        COLOR_EST = "blue"

        # GT trajectory path - use full trajectory for dense visualization
        # Create frame indices for hover tooltips (all frames)
        all_frames = np.arange(0, len(gt_positions_full))

        fig.add_trace(go.Scatter3d(
            x=gt_positions_full[:, 0],
            y=gt_positions_full[:, 1],
            z=gt_positions_full[:, 2],
            mode="lines",
            line=dict(width=5, color=COLOR_GT),
            name="GT path",
            customdata=all_frames,
            hovertemplate="frame: %{customdata}<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>"
        ))

        # Draw orientation arrows
        for arrow_type, color, dirs in [
            ("GT", COLOR_GT, dirs_gt),
            ("EST", COLOR_EST, dirs_est)
        ]:
            hover_color = "red" if color == "red" else "blue"
            hoverlabel_cfg = dict(
                bgcolor=f"rgba({255 if color == 'red' else 0},0,{255 if color == 'blue' else 0},0.85)",
                font=dict(color="white")
            )

            for i in range(len(gt_positions_filtered)):
                if i >= len(dirs):
                    break

                x0, y0, z0 = gt_positions_filtered[i]
                dx, dy, dz = dirs[i]
                label = labels[i] if i < len(labels) else f"frame {i}"

                # Arrow line
                fig.add_trace(go.Scatter3d(
                    x=[x0, x0 + dx],
                    y=[y0, y0 + dy],
                    z=[z0, z0 + dz],
                    mode="lines",
                    line=dict(width=4, color=color),
                    showlegend=False,
                    customdata=[label, label],
                    hovertemplate="frames: %{customdata}<extra></extra>",
                    hoverlabel=hoverlabel_cfg
                ))

                # Arrow cone
                fig.add_trace(go.Cone(
                    x=[x0 + dx], y=[y0 + dy], z=[z0 + dz],
                    u=[dx], v=[dy], w=[dz],
                    anchor="tail",
                    colorscale=[[0, color], [1, color]],
                    sizemode="absolute",
                    sizeref=0.15,
                    showscale=False,
                    customdata=[label],
                    hovertemplate="frames: %{customdata}<extra></extra>",
                    hoverlabel=hoverlabel_cfg
                ))

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X", showbackground=True, gridcolor="lightgray"),
                yaxis=dict(title="Y", showbackground=True, gridcolor="lightgray"),
                zaxis=dict(title="Z", showbackground=True, gridcolor="lightgray"),
                aspectmode="cube",
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.4),
                    up=dict(x=0, y=0, z=1)
                ),
            ),
            title="GT + EST Orientation (3D view)",
            width=1300,
            height=900
        )

        # Save and show
        output_path = self.output_dir / "orientation_plot.html"
        fig.write_html(str(output_path))
        print(f"[INFO] 3D plot saved to: {output_path}")

        fig.show()

        return str(output_path)

    def create_video(self, images_dir, evaluation_results, output_filename="output.mp4", fps=10):
        """
        Create video with GT and EST pose annotations overlaid on frames.

        Args:
            images_dir: Directory containing image files
            evaluation_results: Dict from PoseEvaluator with GT and EST angles
            output_filename: Output video filename
            fps: Frames per second for output video

        Returns:
            str: Path to saved video file
        """
        images_dir = Path(images_dir)
        frames = evaluation_results['frames']
        gt_roll = evaluation_results['gt_roll']
        gt_pitch = evaluation_results['gt_pitch']
        gt_yaw = evaluation_results['gt_yaw']
        est_roll = evaluation_results['est_roll']
        est_pitch = evaluation_results['est_pitch']
        est_yaw = evaluation_results['est_yaw']

        # Get first frame to determine video dimensions
        first_frame_idx = frames[0]
        first_img_path = images_dir / f"{first_frame_idx:06d}.png"
        first_img = cv2.imread(str(first_img_path))

        if first_img is None:
            raise RuntimeError(f"Could not read first image: {first_img_path}")

        height, width = first_img.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = self.output_dir / output_filename
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        print(f"[INFO] Saving video to: {video_path}")

        # Process each frame
        for i, frame_idx in enumerate(frames):
            img_path = images_dir / f"{frame_idx:06d}.png"
            frame = cv2.imread(str(img_path))

            if frame is None:
                print(f"[WARN] Could not read image for frame {frame_idx}, skipping")
                continue

            # Text annotations
            text_frame = f"Frame: {frame_idx}"
            text_gt = f"GT   r={gt_roll[i]:.1f}, p={gt_pitch[i]:.1f}, y={gt_yaw[i]:.1f} deg"
            text_est = f"EST  r={est_roll[i]:.1f}, p={est_pitch[i]:.1f}, y={est_yaw[i]:.1f} deg"

            # Draw frame number (white)
            cv2.putText(
                frame, text_frame,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Draw GT text (red)
            cv2.putText(
                frame, text_gt,
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),  # BGR: red
                2,
                cv2.LINE_AA
            )

            # Draw EST text (blue)
            cv2.putText(
                frame, text_est,
                (30, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),  # BGR: blue
                2,
                cv2.LINE_AA
            )

            writer.write(frame)

        writer.release()
        print(f"[INFO] Video saved to: {video_path}")

        return str(video_path)

    @staticmethod
    def _rpy_to_direction(roll_deg, pitch_deg, yaw_deg):
        """
        Convert roll-pitch-yaw to direction vector.

        Args:
            roll_deg: Roll angle in degrees
            pitch_deg: Pitch angle in degrees
            yaw_deg: Yaw angle in degrees

        Returns:
            np.ndarray: Unit direction vector (3,)
        """
        R = euler_to_rotation_yup(yaw_deg, pitch_deg, roll_deg)

        # Base direction vector (forward in camera frame)
        base = np.array([0, 0, 1], dtype=float)

        # Apply rotation
        direction = R @ base

        # Negate to convert from camera's +Z forward to world's -Z forward (Y-up convention)
        direction = -direction

        return direction / np.linalg.norm(direction)
