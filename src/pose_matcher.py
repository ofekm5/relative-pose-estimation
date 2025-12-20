from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from image_loader import load_image_pair
from feature_extractor import create_feature_extractor, detect_and_compute
from matcher import create_matcher, match_descriptors
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # requires python3-tk installed


class PoseMatcher:
    def __init__(
        self,
        base_dir,
        gt_path,
        feature_method: str = "ORB",
        norm_type: str = "Hamming",
        max_matches: int = 500,
    ):
        """
        base_dir      : simulator base dir (contains images/)
        gt_path       : path to camera_poses.txt
        feature_method: ORB / SIFT / etc. (passed to create_feature_extractor)
        norm_type     : 'Hamming' / 'L2' (passed to create_matcher)
        max_matches   : max matches to keep after sorting by distance
        """

        self.base_dir = Path(base_dir)
        self.gt_path = gt_path
        self.feature_method = feature_method
        self.norm_type = norm_type
        self.max_matches = max_matches

        # --- Load GT once ---
        self.df_gt = pd.read_csv(gt_path, delim_whitespace=True)

        # --- Build K once from a sample image ---
        # take first frame from GT:
        sample_frame = int(self.df_gt["frame"].iloc[0])
        sample_path = self.base_dir / "images" / f"{sample_frame:06d}.png"
        sample_img, _ = load_image_pair(str(sample_path), str(sample_path), to_gray=True)

        self.K = self._build_K(sample_img)

        # --- Build feature extractor + matcher once ---
        self.extractor = create_feature_extractor(self.feature_method)
        self.matcher = create_matcher(norm_type=self.norm_type)

    @staticmethod
    def rotmat_to_ypr_y_up(R):
        """
        Convert rotation matrix R (3x3) into yaw, pitch, roll
        using the convention:
            yaw   = rotation around +Y
            pitch = rotation around +X
            roll  = rotation around +Z

        Returns:
            (roll, pitch, yaw) in radians
        """

        # Extract pitch based on R[2,1] = sin(pitch)
        pitch = np.arcsin(R[2, 1])

        # Handle gimbal lock (pitch = ±90°)
        if abs(R[2, 1]) > 0.9999:
            # roll from elements of column 1
            roll = np.arctan2(-R[1, 2], R[1, 1])
            yaw = 0.0
            return roll, pitch, yaw

        # yaw from first column (cos(pitch) removes coupling)
        yaw = np.arctan2(-R[2, 0], R[0, 0])

        # roll from row/column after removing yaw/pitch influence
        roll = np.arctan2(R[1, 0], R[1, 1])

        return roll, pitch, yaw

    # ------------------------------
    #  Intrinsics (K)
    # ------------------------------
    @staticmethod
    def _build_K(image):
        """
        Build intrinsic matrix K from one sample image,
        using the same scaling logic you used before.
        """
        h, w = image.shape[:2]

        scale_x = w / 960.0
        scale_y = h / 720.0

        fx = 924.82939686 * scale_x
        fy = 920.4766382 * scale_y
        cx = 468.24930789 * scale_x
        cy = 353.65863024 * scale_y

        K = np.array([
            [fx, 0,   cx],
            [0,  fy,  cy],
            [0,  0,   1]
        ])

        return K

    # ------------------------------
    #  YPR → rotation matrix (same convention as your code)
    # ------------------------------
    @staticmethod
    def _ypr_to_rotmat_y_up(roll_deg, pitch_deg, yaw_deg):
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)

        # yaw around +Y
        cy, sy = np.cos(yaw), np.sin(yaw)
        Ry = np.array([
            [ cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy]
        ])

        # pitch around +X
        cx, sx = np.cos(pitch), np.sin(pitch)
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0,  cx, -sx],
            [0.0,  sx,  cx]
        ])

        # roll around +Z
        cz, sz = np.cos(roll), np.sin(roll)
        Rz = np.array([
            [ cz, -sz, 0.0],
            [ sz,  cz, 0.0],
            [0.0, 0.0, 1.0]
        ])

        return Ry @ Rx @ Rz

    # ------------------------------
    #  Low-level: relative pose from images using configured extractor+matcher
    # ------------------------------
    def _compute_relative_pose(self, prev_image, new_image):
        """
        Equivalent to compute_pose_from_images(),
        but uses self.extractor and self.matcher created in __init__.
        Returns:
            R_rel, t, pts1, pts2, matches, mask_pose
        """
        # 1) keypoints & descriptors
        kp1, desc1 = detect_and_compute(prev_image, self.extractor)
        kp2, desc2 = detect_and_compute(new_image, self.extractor)

        if desc1 is None or desc2 is None:
            raise RuntimeError("Could not compute descriptors for one of the images.")

        # 2) matching
        matches = match_descriptors(
            desc1, desc2, self.matcher,
            sort_by_distance=True,
            max_matches=self.max_matches,
        )

        # 3) to Nx2 points
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

        # 4) Essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None:
            raise RuntimeError("Could not estimate Essential matrix.")

        # 5) Recover pose
        _, R_rel, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        return R_rel, t, pts1, pts2, matches, mask_pose

    # ------------------------------
    #  High-level: MATCH API
    # ------------------------------
    def match(self, prev_image, new_image, prev_frame_index):
        """
        High-level API:
        Given:
          - prev_image (numpy)
          - new_image  (numpy)
          - prev_frame_index (int, GT frame index for prev_image)

        Returns:
            (est_azimuth_deg, est_pitch_deg, est_roll_deg)
        where:
            azimuth == yaw (rotation around Y)
        """

        # 1) relative rotation from images
        R_rel, t, pts1, pts2, matches, mask_pose = self._compute_relative_pose(
            prev_image, new_image
        )

        # 2) GT orientation of previous frame
        prev_row = self.df_gt[self.df_gt["frame"] == prev_frame_index].iloc[0]
        prev_roll_deg  = float(prev_row["roll"])
        prev_pitch_deg = float(prev_row["pitch"])
        prev_yaw_deg   = float(prev_row["yaw"])

        # 3) world rotation of previous frame
        R_prev_world = self._ypr_to_rotmat_y_up(prev_roll_deg,
                                                prev_pitch_deg,
                                                prev_yaw_deg)

        # 4) compose to get world rotation of new frame
        #    R_rel = R_cam2^T * R_cam1  ⇒  R_cam2 = R_cam1 @ R_rel^T
        R_new_world = R_prev_world @ R_rel.T

        # 5) back to YPR in your convention
        roll_est, pitch_est, yaw_est = PoseMatcher.rotmat_to_ypr_y_up(R_new_world)

        est_roll_deg  = np.rad2deg(roll_est)
        est_pitch_deg = np.rad2deg(pitch_est)
        est_yaw_deg   = np.rad2deg(yaw_est)

        est_azimuth_deg = est_yaw_deg  # yaw == azimuth

        return est_azimuth_deg, est_pitch_deg, est_roll_deg



if __name__ == '__main__':
    base_dir = "/home/orr/university_projects/relative-pose-estimation/silmulator_data/simple_movement"
    gt_path = base_dir + "/camera_poses.txt"

    matcher = PoseMatcher(
        base_dir=base_dir,
        gt_path=gt_path,
        feature_method="ORB",
        norm_type="Hamming",
        max_matches=500,
    )

    frame1 = 230
    frame2 = 240


    img1_path = Path(base_dir) / "images" / f"{frame1:06d}.png"
    img2_path = Path(base_dir) / "images" / f"{frame2:06d}.png"

    img1, img2 = load_image_pair(str(img1_path), str(img2_path), to_gray=True)

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[:200]


    # כל נקודות ORB (לפני match)
    all_pts1 = np.array([kp.pt for kp in kp1])
    all_pts2 = np.array([kp.pt for kp in kp2])

    matched_pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    matched_pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
    az, pitch, roll = matcher.match(img1, img2, prev_frame_index=frame1)

    print("est azimuth:", az)
    print("est pitch  :", pitch)
    print("est roll   :", roll)
    gt_row = matcher.df_gt[matcher.df_gt["frame"] == frame2].iloc[0]
    gt_yaw   = float(gt_row["yaw"])
    gt_pitch = float(gt_row["pitch"])
    gt_roll  = float(gt_row["roll"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(img1, cmap="gray"); ax[0].set_title(f"Frame {frame1}"); ax[0].axis("off")
    ax[1].imshow(img2,cmap="gray"); ax[1].set_title(f"Frame {frame2}"); ax[1].axis("off")

    ax[0].scatter(all_pts1[:, 0], all_pts1[:, 1], s=8, c="red", label="ORB (before match)")
    ax[1].scatter(all_pts2[:, 0], all_pts2[:, 1], s=8, c="red")

    ax[0].scatter(matched_pts1[:, 0], matched_pts1[:, 1], s=20, c="blue", label="After match")
    ax[1].scatter(matched_pts2[:, 0], matched_pts2[:, 1], s=20, c="blue")

    ax[0].legend(loc="lower right")

    fig.suptitle("Two Frames + ESM + GT (last frame)")
    fig.text(
        0.01, 0.01,
        f"ESM ([deg]: yaw={az:.3f}, pitch={pitch:.3f}, roll={roll:.3f}\n"
        f"GT  [deg]: yaw={gt_yaw:.3f}, pitch={gt_pitch:.3f}, roll={gt_roll:.3f}\n",
        family="monospace",
        fontsize=9,
        va="bottom"
    )
    plt.tight_layout()
    plt.show()