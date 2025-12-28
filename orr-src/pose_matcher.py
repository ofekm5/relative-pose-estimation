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

        # ---- NEW: line-assisted filtering ----
        use_lines: bool = False,
        line_refine: int = cv2.LSD_REFINE_STD,
        line_dilate_px: int = 2,
        min_points_after_line_filter: int = 40,   # אם נשאר פחות מזה - עושים fallback
    ):
        """
        base_dir      : simulator base dir (contains images/)
        gt_path       : path to camera_poses.txt
        feature_method: ORB / SIFT / etc. (passed to create_feature_extractor)
        norm_type     : 'Hamming' / 'L2' (passed to create_matcher)
        max_matches   : max matches to keep after sorting by distance

        use_lines     : אם True, נשתמש בקווי LSD כדי לסנן התאמות:
                        נשאיר רק matches שהנקודות שלהן קרובות לקווים בשתי התמונות
        line_dilate_px: כמה "עובי" לתת לקווים במסכה (כדי לתפוס נקודות קרובות לקו)
        min_points_after_line_filter:
                        אם אחרי הסינון נשאר מעט מדי נקודות, חוזרים ל-matches המקוריים
        """

        self.base_dir = Path(base_dir)
        self.gt_path = gt_path
        self.feature_method = feature_method
        self.norm_type = norm_type
        self.max_matches = max_matches

        # ---- NEW: line-assisted parameters ----
        self.use_lines = use_lines
        self.line_refine = line_refine
        self.line_dilate_px = int(line_dilate_px)
        self.min_points_after_line_filter = int(min_points_after_line_filter)

        # --- Load GT once ---
        self.df_gt = pd.read_csv(gt_path, delim_whitespace=True)

        # --- Build K once from a sample image ---
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
        pitch = np.arcsin(R[2, 1])

        if abs(R[2, 1]) > 0.9999:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            yaw = 0.0
            return roll, pitch, yaw

        yaw = np.arctan2(-R[2, 0], R[0, 0])
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
    #  NEW: LSD line mask (for filtering matches)
    # ------------------------------
    @staticmethod
    def _lsd_line_mask(gray: np.ndarray, refine=cv2.LSD_REFINE_STD, dilate_px: int = 2):
        """
        יוצר מסכת פיקסלים של קווים (0/255) ע"י LSD.
        dilate_px נותן "עובי" לקווים כדי לתפוס נקודות ORB שקרובות לקו.
        """
        if gray.ndim != 2:
            raise ValueError("LSD expects a grayscale image (H,W).")

        lsd = cv2.createLineSegmentDetector(refine)
        lines, _, _, _ = lsd.detect(gray)

        mask = np.zeros_like(gray, dtype=np.uint8)

        if lines is not None and len(lines) > 0:
            # lines shape: (N,1,4) => x1,y1,x2,y2
            for l in lines:
                x1, y1, x2, y2 = l[0]
                x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                cv2.line(mask, (x1i, y1i), (x2i, y2i), 255, thickness=1)

            if dilate_px > 0:
                k = 2 * dilate_px + 1
                kernel = np.ones((k, k), dtype=np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

        return mask, lines

    # ------------------------------
    #  Low-level: relative pose from images using configured extractor+matcher
    # ------------------------------
    def _compute_relative_pose(self, prev_image, new_image):
        """
        Returns:
            R_rel, t, pts1, pts2, matches, mask_pose
        """

        # 1) keypoints & descriptors
        kp1, desc1 = detect_and_compute(prev_image, self.extractor)
        kp2, desc2 = detect_and_compute(new_image, self.extractor)

        if desc1 is None or desc2 is None:
            raise RuntimeError("Could not compute descriptors for one of the images.")

        # 2) matching (as before)
        matches = match_descriptors(
            desc1, desc2, self.matcher,
            sort_by_distance=True,
            max_matches=self.max_matches,
        )

        if len(matches) < 8:
            raise RuntimeError("Not enough matches to estimate Essential matrix.")

        # 3) OPTIONAL: line-assisted filtering on matches
        #    נשאיר matches שהנקודה בשתי התמונות יושבת על מסכת הקווים
        if self.use_lines:
            mask1, _ = self._lsd_line_mask(prev_image, refine=self.line_refine, dilate_px=self.line_dilate_px)
            mask2, _ = self._lsd_line_mask(new_image, refine=self.line_refine, dilate_px=self.line_dilate_px)

            H1, W1 = mask1.shape[:2]
            H2, W2 = mask2.shape[:2]

            filtered = []
            for m in matches:
                x1, y1 = kp1[m.queryIdx].pt
                x2, y2 = kp2[m.trainIdx].pt

                x1i, y1i = int(round(x1)), int(round(y1))
                x2i, y2i = int(round(x2)), int(round(y2))

                if not (0 <= x1i < W1 and 0 <= y1i < H1 and 0 <= x2i < W2 and 0 <= y2i < H2):
                    continue

                if mask1[y1i, x1i] > 0 and mask2[y2i, x2i] > 0:
                    filtered.append(m)

            # אם סיננו יותר מדי - עושים fallback
            if len(filtered) >= max(8, self.min_points_after_line_filter):
                matches = filtered
            # else: נשארים עם matches המקוריים

        # 4) to Nx2 points
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

        if len(pts1) < 8:
            raise RuntimeError("Not enough points after filtering to estimate Essential matrix.")

        # 5) Essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None:
            raise RuntimeError("Could not estimate Essential matrix.")

        # 6) Recover pose
        _, R_rel, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        return R_rel, t, pts1, pts2, matches, mask_pose

    # ------------------------------
    #  High-level: MATCH API
    # ------------------------------
    def match(self, prev_image, new_image, prev_frame_index):
        """
        Returns:
            (est_azimuth_deg, est_pitch_deg, est_roll_deg)
        where azimuth == yaw (rotation around Y)
        """

        # 1) relative rotation from images
        R_rel, t, pts1, pts2, matches, mask_pose = self._compute_relative_pose(
            prev_image, new_image
        )

        # 2) GT orientation of previous frame
        prev_row = self.df_gt[self.df_gt["frame"] == prev_frame_index].iloc[0]
        prev_roll_deg = float(prev_row["roll"])
        prev_pitch_deg = float(prev_row["pitch"])
        prev_yaw_deg = float(prev_row["yaw"])

        # 3) world rotation of previous frame
        R_prev_world = self._ypr_to_rotmat_y_up(
            prev_roll_deg,
            prev_pitch_deg,
            prev_yaw_deg
        )

        # 4) compose to get world rotation of new frame
        #    R_rel = R_cam2^T * R_cam1  ⇒  R_cam2 = R_cam1 @ R_rel^T
        R_new_world = R_prev_world @ R_rel


        # 5) back to YPR in your convention
        roll_est, pitch_est, yaw_est = PoseMatcher.rotmat_to_ypr_y_up(R_new_world)

        est_roll_deg = np.rad2deg(roll_est)
        est_pitch_deg = np.rad2deg(pitch_est)
        est_yaw_deg = np.rad2deg(yaw_est)

        est_azimuth_deg = est_yaw_deg  # yaw == azimuth

        return est_azimuth_deg, est_pitch_deg, est_roll_deg


if __name__ == '__main__':
    base_dir = "/home/orr/university_projects/relative-pose-estimation/silmulator_data/simple_movement"
    gt_path = base_dir + "/camera_poses.txt"

    # ---- NEW: use_lines=True כדי להפעיל סינון ע"י קווים ----
    matcher = PoseMatcher(
        base_dir=base_dir,
        gt_path=gt_path,
        feature_method="ORB",
        norm_type="Hamming",
        max_matches=500,
        use_lines=True,             # <-- כאן
        line_dilate_px=2,
        min_points_after_line_filter=40,
    )

    frame1 = 230
    frame2 = 235

    img1_path = Path(base_dir) / "images" / f"{frame1:06d}.png"
    img2_path = Path(base_dir) / "images" / f"{frame2:06d}.png"

    img1, img2 = load_image_pair(str(img1_path), str(img2_path), to_gray=True)

    # ORB just for plotting points (as your previous demo)
    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[:200]

    all_pts1 = np.array([kp.pt for kp in kp1])
    all_pts2 = np.array([kp.pt for kp in kp2])

    matched_pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    matched_pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

    az, pitch, roll = matcher.match(img1, img2, prev_frame_index=frame1)

    print("est azimuth:", az)
    print("est pitch  :", pitch)
    print("est roll   :", roll)

    gt_row = matcher.df_gt[matcher.df_gt["frame"] == frame2].iloc[0]
    gt_yaw = float(gt_row["yaw"])
    gt_pitch = float(gt_row["pitch"])
    gt_roll = float(gt_row["roll"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(img1, cmap="gray"); ax[0].set_title(f"Frame {frame1}"); ax[0].axis("off")
    ax[1].imshow(img2, cmap="gray"); ax[1].set_title(f"Frame {frame2}"); ax[1].axis("off")

    ax[0].scatter(all_pts1[:, 0], all_pts1[:, 1], s=8, c="red", label="ORB (before match)")
    ax[1].scatter(all_pts2[:, 0], all_pts2[:, 1], s=8, c="red")

    ax[0].scatter(matched_pts1[:, 0], matched_pts1[:, 1], s=20, c="blue", label="After match (demo)")
    ax[1].scatter(matched_pts2[:, 0], matched_pts2[:, 1], s=20, c="blue")

    ax[0].legend(loc="lower right")

    fig.suptitle("Two Frames + Pose Estimation + GT (last frame)")
    fig.text(
        0.01, 0.01,
        f"EST ([deg]: yaw={az:.3f}, pitch={pitch:.3f}, roll={roll:.3f})\n"
        f"GT  ([deg]: yaw={gt_yaw:.3f}, pitch={gt_pitch:.3f}, roll={gt_roll:.3f})\n",
        family="monospace",
        fontsize=9,
        va="bottom"
    )
    plt.tight_layout()
    plt.show()
