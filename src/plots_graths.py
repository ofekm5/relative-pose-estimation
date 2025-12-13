import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from .image_loader import load_image_pair
from .pose_matcher import PoseMatcher
import cv2

class PosePlotter:

    def __init__(self, base_path, step=15, arrow_scale=0.3):
        self.base_path = Path(base_path)
        self.step = step
        self.arrow_scale = arrow_scale

        self.gt_path = self.base_path / "camera_poses.txt"
        self.images_dir = self.base_path / "images"

        self.df = None
        self.df_sub = None
        self.matcher = None

        # results
        self.est_roll = []
        self.est_pitch = []
        self.est_yaw = []
        self.frames_est = []

        self.results_dir = self.base_path / "results"
        self.results_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # RPY → direction vector
    # --------------------------------------------------------
    @staticmethod
    def rpy_to_dir(roll_deg, pitch_deg, yaw_deg):
        r = np.deg2rad(roll_deg)
        p = np.deg2rad(pitch_deg)
        y = np.deg2rad(yaw_deg)

        Ry = np.array([
            [ np.cos(y), 0, np.sin(y)],
            [ 0, 1, 0 ],
            [-np.sin(y), 0, np.cos(y)]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(p), -np.sin(p)],
            [0, np.sin(p),  np.cos(p)]
        ])

        Rz = np.array([
            [ np.cos(r), -np.sin(r), 0],
            [ np.sin(r),  np.cos(r), 0],
            [ 0, 0, 1]
        ])

        R = Ry @ Rx @ Rz
        v = R @ np.array([0, 0, 1])
        return -v   # flip to match your convention

    # --------------------------------------------------------
    # Load GT Data
    # --------------------------------------------------------
    def load_gt(self):
        self.df = pd.read_csv(self.gt_path, delim_whitespace=True)
        self.df_sub = self.df[self.df["frame"] % self.step == 0]

        self.matcher = PoseMatcher(
            base_dir=str(self.base_path),
            gt_path=str(self.gt_path),
            feature_method="ORB",
            norm_type="Hamming",
            max_matches=500,
        )

    # --------------------------------------------------------
    # Compute estimated RPY between frames
    # --------------------------------------------------------
    def compute_estimated(self):
        for i in range(len(self.df_sub) - 1):
            f0 = int(self.df_sub.iloc[i]["frame"])
            f1 = int(self.df_sub.iloc[i + 1]["frame"])

            img1 = self.images_dir / f"{f0:06d}.png"
            img2 = self.images_dir / f"{f1:06d}.png"

            im1, im2 = load_image_pair(str(img1), str(img2), to_gray=True)

            az, pitch, roll = self.matcher.match(im1, im2, prev_frame_index=f0)

            self.frames_est.append(f1)
            self.est_roll.append(roll)
            self.est_pitch.append(pitch)
            self.est_yaw.append(az)

        self.est_roll = np.array(self.est_roll)
        self.est_pitch = np.array(self.est_pitch)
        self.est_yaw = np.array(self.est_yaw)

    # --------------------------------------------------------
    # Build orientation vectors (GT + EST)
    # --------------------------------------------------------
    def build_vectors(self):

        origins = self.df_sub[["x", "y", "z"]].to_numpy()

        dirs_gt = []
        labels = []

        for _, row in self.df_sub.iterrows():
            d = self.rpy_to_dir(row["roll"], row["pitch"], row["yaw"])
            dirs_gt.append(d * self.arrow_scale)

            start = int(row["frame"])
            labels.append(f"{start}-{start + self.step}")

        dirs_gt = np.array(dirs_gt)

        # Estimated dirs
        dirs_est = []
        for r, p, y in zip(self.est_roll, self.est_pitch, self.est_yaw):
            d = self.rpy_to_dir(r, p, y)
            dirs_est.append(d * self.arrow_scale)

        # first arrow copies GT (same as original code)
        dirs_est = [dirs_gt[0]] + dirs_est
        dirs_est = np.array(dirs_est)

        return origins, dirs_gt, dirs_est, labels

    # --------------------------------------------------------
    # Plot everything
    # --------------------------------------------------------
    def plot(self, origins, dirs_gt, dirs_est, labels):

        fig = go.Figure()
        COLOR_GT = "red"
        COLOR_EST = "blue"

        # GT path
        fig.add_trace(go.Scatter3d(
            x=self.df["x"],
            y=self.df["y"],
            z=self.df["z"],
            mode="lines",
            line=dict(width=5, color=COLOR_GT),
            name="GT path",
            customdata=self.df["frame"],
            hovertemplate="frame: %{customdata}<br>"
                          "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>"
        ))

        # Draw arrows
        for arr, color, dirs in [
            ("GT", COLOR_GT, dirs_gt),
            ("EST", COLOR_EST, dirs_est)
        ]:

            # צבע Hover
            hover_bg = color  # red / blue
            hoverlabel_cfg = dict(
                bgcolor=f"rgba({255 if color == 'red' else 0},0,{255 if color == 'blue' else 0},0.85)",
                font=dict(color="white")
            )

            for i in range(len(origins)):
                x0, y0, z0 = origins[i]
                dx, dy, dz = dirs[i]
                label = labels[i]

                # line
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

                # cone
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

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X", showbackground=True, gridcolor="lightgray"),
                yaxis=dict(title="Y", showbackground=True, gridcolor="lightgray"),
                zaxis=dict(title="Z", showbackground=True, gridcolor="lightgray"),
                aspectmode="cube",  # כל הצירים באותו סקייל -> מרגיש יותר תלת מימד
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.4),  # מאיפה המצלמה "מסתכלת"
                    up=dict(x=0, y=0, z=1)  # כיוון "למעלה" (Z למעלה)
                ),
            ),
            title="GT + EST Orientation (3D view)"
        )
        output_html = self.results_dir / "orientation_plot.html"
        fig.write_html(str(output_html))
        print(f"[INFO] 3D plot saved to: {output_html}")

        fig.show()  # OPTIONAL – keep if you still want to see it liv

    # --------------------------------------------------------
    # MAIN ENTRY POINT
    # --------------------------------------------------------
    def run(self):
        self.load_gt()
        self.compute_estimated()
        origins, dirs_gt, dirs_est, labels = self.build_vectors()
        self.plot(origins, dirs_gt, dirs_est, labels)



    def make_video(self, output_file, fps=10):
        """
        יוצר סרט מהתמונות ב-images/, ועל כל פריים כותב:
        - מספר פריים
        - זוויות GT (roll, pitch, yaw)
        - זוויות EST (roll, pitch, yaw)
        """

        # לוודא שהנתונים מחושבים
        if self.df is None or self.df_sub is None or len(self.est_yaw) == 0:
            self.load_gt()
            self.compute_estimated()

        # נבדוק פריים ראשון שיש לו EST (מתחיל מ-i=1)
        first_frame_idx = int(self.df_sub.iloc[1]["frame"])
        first_img_path = self.images_dir / f"{first_frame_idx:06d}.png"
        first_img = cv2.imread(str(first_img_path))

        if first_img is None:
            raise RuntimeError(f"Could not read first image: {first_img_path}")

        height, width = first_img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = self.results_dir / output_file
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        print(f"[INFO] Saving video to: {video_path}")

        # est_*[i-1] מתאים ל df_sub.iloc[i]
        for i in range(1, len(self.df_sub)):
            row = self.df_sub.iloc[i]
            frame_idx = int(row["frame"])

            img_path = self.images_dir / f"{frame_idx:06d}.png"
            frame = cv2.imread(str(img_path))

            if frame is None:
                print(f"[WARN] Could not read image for frame {frame_idx}, skipping")
                continue

            # --- GT angles ---
            gt_roll  = float(row["roll"])
            gt_pitch = float(row["pitch"])
            gt_yaw   = float(row["yaw"])

            # --- EST angles (מה-PoseMatcher) ---
            est_roll  = float(self.est_roll[i - 1])
            est_pitch = float(self.est_pitch[i - 1])
            est_yaw   = float(self.est_yaw[i - 1])

            # --- Text lines ---
            text_frame = f"Frame: {frame_idx}"
            text_gt    = f"GT   r={gt_roll:.1f}, p={gt_pitch:.1f}, y={gt_yaw:.1f} deg"
            text_est   = f"EST  r={est_roll:.1f}, p={est_pitch:.1f}, y={est_yaw:.1f} deg"

            # --- Draw text on frame ---
            # Frame number (white)
            cv2.putText(
                frame, text_frame,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # GT (red)
            cv2.putText(
                frame, text_gt,
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),   # BGR: red
                2,
                cv2.LINE_AA
            )

            # EST (blue)
            cv2.putText(
                frame, text_est,
                (30, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),   # BGR: blue
                2,
                cv2.LINE_AA
            )

            writer.write(frame)

        writer.release()
        print(f"[INFO] Video saved to: {video_path}")



if __name__ == '__main__':
    base_path = "/home/orr/university_projects/relative-pose-estimation/silmulator_data/simple_movement"

    plotter = PosePlotter(base_path)

    # אם אתה רוצה גם פלוט 3D:
    plotter.run()

    # ועכשיו סרט:
    plotter.make_video(output_file="yaw_debug.mp4", fps=5)