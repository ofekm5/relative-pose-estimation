from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import os
import itertools
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
        K,
        source,
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
        # VP refinement defaults
        self.use_vp_refinement = True
        self.vp_max_lines = 120
        self.vp_max_pairs = 3000
        self.vp_acc_min = 8e5
        self.vp_vp2_min = 8000.0
        self.vp_iters = 12
        self.vp_lm_lambda = 1e-2
        self.vp_cost_improve_eps = 1e-3
        self.K = K
        # --- Load GT once ---
        if source =="simulator":
            self.df_gt = pd.read_csv(gt_path, sep=r"\s+")
        else:
            self.df_gt = pd.read_csv(gt_path)

        # --- Build K once from a sample image ---
        # take first frame from GT:
        sample_frame = int(self.df_gt["frame"].iloc[0])
        sample_path = self.base_dir / "images" / f"{sample_frame:06d}.png"
        sample_img, _ = load_image_pair(str(sample_path), str(sample_path), to_gray=True)



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
# ============================================================
#  VP refinement (LSD lines -> VP voting -> Manhattan dirs)
#  Inspired by VP-SLAM (arXiv:2210.12756):
#  weight ~ |l0||l1|sin(2θ), polar grid voting on Gaussian sphere.
# ============================================================

    @staticmethod
    def _detect_lsd_lines(gray: np.ndarray) -> np.ndarray:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines, _, _, _ = lsd.detect(gray)
        if lines is None:
            return np.zeros((0, 4), dtype=np.float64)
        return lines.reshape(-1, 4).astype(np.float64)

    @staticmethod
    def _line_angle_and_len(line: np.ndarray):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy) + 1e-9)
        angle = float(np.arctan2(dy, dx))
        return angle, length

    @staticmethod
    def _hom_line_from_segment(line: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = line
        p1 = np.array([x1, y1, 1.0], dtype=np.float64)
        p2 = np.array([x2, y2, 1.0], dtype=np.float64)
        l = np.cross(p1, p2)
        n = np.linalg.norm(l[:2]) + 1e-12
        return l / n

    @staticmethod
    def _vp_dir_from_image_point(vp_xy, K: np.ndarray) -> np.ndarray:
        x, y = float(vp_xy[0]), float(vp_xy[1])
        v = np.array([x, y, 1.0], dtype=np.float64)
        d = np.linalg.inv(K) @ v
        d = d / (np.linalg.norm(d) + 1e-12)
        # half-sphere convention
        if d[2] < 0:
            d = -d
        return d

    @staticmethod
    def _dir_to_grid_idx(d: np.ndarray, n_lat=90, n_lon=360):
        # half sphere z>0
        lat = np.arctan2(np.hypot(d[0], d[1]), d[2])  # 0..pi/2
        lon = np.arctan2(d[1], d[0])                  # -pi..pi
        lat_deg = np.rad2deg(lat)
        lon_deg = (np.rad2deg(lon) + 360.0) % 360.0
        lat_i = int(np.clip(lat_deg, 0, n_lat - 1))
        lon_i = int(np.clip(lon_deg, 0, n_lon - 1))
        return lat_i, lon_i

    @staticmethod
    def estimate_manhattan_dirs_from_image(gray: np.ndarray,
                                           K: np.ndarray,
                                           max_lines=200,
                                           n_lat=90,
                                           n_lon=360,
                                           max_pairs=15000,
                                           rng_seed=0):
        '''
        Returns:
            Delta: (3,3) columns are δ_k (camera-frame Manhattan directions)
            ok: bool
            dbg: dict with num_lines, acc_max, lines_used, vp2_score
        '''
        lines = PoseMatcher._detect_lsd_lines(gray)
        dbg = {"num_lines": int(lines.shape[0])}
        if lines.shape[0] < 10:
            return None, False, dbg

        lens_all = np.array([PoseMatcher._line_angle_and_len(l)[1] for l in lines])
        idx = np.argsort(-lens_all)[:min(max_lines, len(lines))]
        lines = lines[idx]
        lens = lens_all[idx]

        hlines = np.array([PoseMatcher._hom_line_from_segment(l) for l in lines])
        angles = np.array([PoseMatcher._line_angle_and_len(l)[0] for l in lines])

        acc = np.zeros((n_lat, n_lon), dtype=np.float64)

        m = len(lines)
        total_pairs = m * (m - 1) // 2
        if total_pairs <= max_pairs:
            pairs = list(itertools.combinations(range(m), 2))
        else:
            rng = np.random.default_rng(rng_seed)
            pairs = []
            for _ in range(max_pairs):
                i = int(rng.integers(0, m))
                j = int(rng.integers(0, m))
                if i != j:
                    if i > j:
                        i, j = j, i
                    pairs.append((i, j))

        for i, j in pairs:
            li = hlines[i]
            lj = hlines[j]
            vp = np.cross(li, lj)
            if abs(vp[2]) < 1e-9:
                continue
            vp_xy = (vp[0] / vp[2], vp[1] / vp[2])

            theta = abs(angles[i] - angles[j])
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            theta = abs(theta)

            w = float(lens[i] * lens[j] * abs(np.sin(2.0 * theta)))
            if w <= 0:
                continue

            d = PoseMatcher._vp_dir_from_image_point(vp_xy, K)
            lat_i, lon_i = PoseMatcher._dir_to_grid_idx(d, n_lat=n_lat, n_lon=n_lon)
            acc[lat_i, lon_i] += w

        acc_max = float(np.max(acc))
        dbg["acc_max"] = acc_max
        dbg["lines_used"] = int(m)
        if acc_max <= 0:
            return None, False, dbg

        # VP1: argmax cell -> direction
        lat1, lon1 = np.unravel_index(np.argmax(acc), acc.shape)
        lat1_rad = np.deg2rad(lat1 + 0.5)
        lon1_rad = np.deg2rad(lon1 + 0.5)
        v1 = np.array([
            np.sin(lat1_rad) * np.cos(lon1_rad),
            np.sin(lat1_rad) * np.sin(lon1_rad),
            np.cos(lat1_rad)
        ], dtype=np.float64)
        v1 /= (np.linalg.norm(v1) + 1e-12)

        # VP2 on great circle ⟂ v1 (360 candidates)
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, v1)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        a = np.cross(v1, tmp); a /= (np.linalg.norm(a) + 1e-12)
        b = np.cross(v1, a);   b /= (np.linalg.norm(b) + 1e-12)

        best_score = -1.0
        v2 = None
        for deg in range(360):
            ang = np.deg2rad(deg)
            cand = np.cos(ang) * a + np.sin(ang) * b
            cand /= (np.linalg.norm(cand) + 1e-12)
            lat_i, lon_i = PoseMatcher._dir_to_grid_idx(cand, n_lat=n_lat, n_lon=n_lon)
            s = acc[lat_i, lon_i]
            if s > best_score:
                best_score = float(s)
                v2 = cand

        dbg["vp2_score"] = float(best_score)
        if v2 is None or best_score <= 0:
            return None, False, dbg

        v3 = np.cross(v1, v2)
        v3 /= (np.linalg.norm(v3) + 1e-12)
        v2 = np.cross(v3, v1)
        v2 /= (np.linalg.norm(v2) + 1e-12)

        Delta = np.stack([v1, v2, v3], axis=1)
        return Delta, True, dbg

    # ============================================================
    #  Paper-style absolute rotation optimization on SO(3)
    #  E(R) = sum_k arccos( δ_k · (R d_k) )
    # ============================================================

    @staticmethod
    def so3_exp(w: np.ndarray) -> np.ndarray:
        R, _ = cv2.Rodrigues(w.reshape(3, 1))
        return R

    @staticmethod
    def vp_cost(R_iw: np.ndarray, Delta_cam: np.ndarray, D_world: np.ndarray) -> float:
        cost = 0.0
        for k in range(3):
            delta = Delta_cam[:, k]
            d = D_world[:, k]
            u = R_iw @ d
            s = float(np.clip(delta @ u, -1.0, 1.0))
            cost += float(np.arccos(s))
        return float(cost)

    @staticmethod
    def optimize_absolute_rotation_from_vps(R_init: np.ndarray,
                                           Delta_cam: np.ndarray,
                                           D_world: np.ndarray,
                                           iters=12,
                                           lm_lambda=1e-2) -> np.ndarray:
        R = R_init.copy()
        for _ in range(iters):
            r_list = []
            J_list = []

            for k in range(3):
                delta = Delta_cam[:, k]
                d = D_world[:, k]
                u = R @ d

                s = float(np.clip(delta @ u, -1.0, 1.0))
                e = float(np.arccos(s))
                r_list.append(e)

                denom = np.sqrt(max(1e-12, 1.0 - s * s))
                cross = np.cross(delta, u)          # (3,)
                J = -(1.0 / denom) * cross          # (3,)
                J_list.append(J.reshape(1, 3))

            r = np.array(r_list, dtype=np.float64).reshape(3, 1)
            J = np.vstack(J_list)

            H = J.T @ J + lm_lambda * np.eye(3)
            g = J.T @ r

            try:
                dw = -np.linalg.solve(H, g).reshape(3,)
            except np.linalg.LinAlgError:
                break

            R = PoseMatcher.so3_exp(dw) @ R
            if np.linalg.norm(dw) < 1e-7:
                break

        return R

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
        # 4) compose to get world rotation of new frame (Option B, validated)
        R_new_init = R_prev_world @ R_rel
        R_new_world = R_new_init.copy()

        # 4b) VP refinement (closest-to-paper), with strict gating + cost gate
        if getattr(self, "use_vp_refinement", True):
            # compute / refresh K if image shape changed

            Delta_prev, ok1, dbg1 = PoseMatcher.estimate_manhattan_dirs_from_image(
                prev_image, self.K,
                max_lines=getattr(self, "vp_max_lines", 200),
                max_pairs=getattr(self, "vp_max_pairs", 15000),
                rng_seed=0
            )
            Delta_new, ok2, dbg2 = PoseMatcher.estimate_manhattan_dirs_from_image(
                new_image, self.K,
                max_lines=getattr(self, "vp_max_lines", 200),
                max_pairs=getattr(self, "vp_max_pairs", 15000),
                rng_seed=1
            )

            # Reliability gate (your logs showed new-frame VPs can be weak)
            ACC_MIN = getattr(self, "vp_acc_min", 8e5)
            VP2_MIN = getattr(self, "vp_vp2_min", 8000.0)
            vp_good_prev = ok1 and (dbg1.get("acc_max", 0.0) >= ACC_MIN) and (dbg1.get("vp2_score", 0.0) >= VP2_MIN)
            vp_good_new  = ok2 and (dbg2.get("acc_max", 0.0) >= ACC_MIN) and (dbg2.get("vp2_score", 0.0) >= VP2_MIN)

            if vp_good_prev and vp_good_new:
                # δ ~ R d  =>  d = R_prev^T δ_prev  (build world Manhattan dirs from prev frame)
                D_world = R_prev_world.T @ Delta_prev
                cost_init = PoseMatcher.vp_cost(R_new_init, Delta_new, D_world)

                R_opt = PoseMatcher.optimize_absolute_rotation_from_vps(
                    R_init=R_new_init,
                    Delta_cam=Delta_new,
                    D_world=D_world,
                    iters=getattr(self, "vp_iters", 12),
                    lm_lambda=getattr(self, "vp_lm_lambda", 1e-2),
                )
                cost_opt = PoseMatcher.vp_cost(R_opt, Delta_new, D_world)

                # Accept only if VP cost improves
                eps = getattr(self, "vp_cost_improve_eps", 1e-3)
                if cost_opt < cost_init - eps:
                    R_new_world = R_opt
            # else: keep points-only

        # 5) back to YPR in your convention
        roll_est, pitch_est, yaw_est = PoseMatcher.rotmat_to_ypr_y_up(R_new_world)

        est_roll_deg  = np.rad2deg(roll_est)
        est_pitch_deg = np.rad2deg(pitch_est)
        est_yaw_deg   = np.rad2deg(yaw_est)

        est_azimuth_deg = est_yaw_deg  # yaw == azimuth

        return est_azimuth_deg, est_pitch_deg, est_roll_deg




# for silmulator



def build_K_simulator(image):
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


CALIB_W, CALIB_H = 2000, 1126  # resolution used during calibration

CALIB_NPZ = "../phone_camera/camera_calibration_code/calibration_filtered.npz"
def build_K_Phone(image):
    """
    Build intrinsic matrix K from one sample image,
    using the same scaling logic you used before.
    """
    if not os.path.isfile(CALIB_NPZ):
        raise RuntimeError(f"Calibration file not found: {CALIB_NPZ}")
    data = np.load(CALIB_NPZ)
    if "K" not in data.files:
        raise RuntimeError(f"npz must contain 'K'. Found: {data.files}")
    K = data["K"].astype(np.float64)

    h, w = image.shape[:2]
    # SCALE THE K
    sx = w / float(CALIB_W)
    sy = h / float(CALIB_H)
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy

    return K2



def build_good_K(image):
    h, w = image.shape[:2]

    fx = fy = 500
    cx = w / 2
    cy = h / 2
    K_test = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0, 0, 1]])
    return K_test

def run_simulator():
    base_dir = "../silmulator_data/simple_movement"
    gt_path = base_dir + "/camera_poses.txt"

    frame1 = 0
    frame2 = 1
    img1_path = Path(base_dir) / "images" / f"{frame1:06d}.png"
    img2_path = Path(base_dir) / "images" / f"{frame2:06d}.png"

    img1, img2 = load_image_pair(str(img1_path), str(img2_path), to_gray=True)
    k = build_K_simulator(img1)
    matcher = PoseMatcher(
        base_dir=base_dir,
        gt_path=gt_path,
        K=k,
        source ="simulator",
        feature_method="ORB",
        norm_type="Hamming",
        max_matches=500)





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
    parallax = np.linalg.norm(matched_pts1 - matched_pts2, axis=1)
    print(np.mean(parallax), np.std(parallax))
    az, pitch, roll = matcher.match(img1, img2, prev_frame_index=frame1)

    print("est azimuth:", az)
    print("est pitch  :", pitch)
    print("est roll   :", roll)
    gt_row = matcher.df_gt[matcher.df_gt["frame"] == frame2].iloc[0]
    gt_yaw = float(gt_row["yaw"])
    gt_pitch = float(gt_row["pitch"])
    gt_roll = float(gt_row["roll"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(img1, cmap="gray");
    ax[0].set_title(f"Frame {frame1}");
    ax[0].axis("off")
    ax[1].imshow(img2, cmap="gray");
    ax[1].set_title(f"Frame {frame2}");
    ax[1].axis("off")

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


def run_RealPhoneCameraRightToLeft():
    base_dir = "../phone_camera/from_right_to_left"
    gt_path = base_dir + "/tag0_pose_filtered.csv"
    frame1 = 0
    frame2 = 25
    img1_path = Path(base_dir) / "images" / f"{frame1:06d}.png"
    img2_path = Path(base_dir) / "images" / f"{frame2:06d}.png"

    img1, img2 = load_image_pair(str(img1_path), str(img2_path), to_gray=True)
    k = build_K_Phone(img1)
    matcher = PoseMatcher(
        base_dir=base_dir,
        gt_path=gt_path,
        K=k,
        source="phone",
        feature_method="ORB",
        norm_type="Hamming",
        max_matches=500,
    )



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
    parallax = np.linalg.norm(matched_pts1 - matched_pts2, axis=1)
    print(np.mean(parallax), np.std(parallax))
    az, pitch, roll = matcher.match(img1, img2, prev_frame_index=frame1)

    print("est azimuth:", az)
    print("est pitch  :", pitch)
    print("est roll   :", roll)
    gt_row = matcher.df_gt[matcher.df_gt["frame"] == frame2].iloc[0]
    gt_yaw = float(gt_row["yaw"])
    gt_pitch = float(gt_row["pitch"])
    gt_roll = float(gt_row["roll"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(img1, cmap="gray");
    ax[0].set_title(f"Frame {frame1}");
    ax[0].axis("off")
    ax[1].imshow(img2, cmap="gray");
    ax[1].set_title(f"Frame {frame2}");
    ax[1].axis("off")

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


def run_RealPhoneCameraForward():
    base_dir = "../phone_camera/forward_with_stuff"
    gt_path = base_dir + "/tag0_pose_filtered.csv"
    frame1 = 0
    frame2 = 20
    img1_path = Path(base_dir) / "images" / f"{frame1:06d}.png"
    img2_path = Path(base_dir) / "images" / f"{frame2:06d}.png"

    img1, img2 = load_image_pair(str(img1_path), str(img2_path), to_gray=True)
    k = build_good_K(img1)
    matcher = PoseMatcher(
        base_dir=base_dir,
        gt_path=gt_path,
        K=k,
        source="phone",
        feature_method="ORB",
        norm_type="Hamming",
        max_matches=500,
    )



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
    parallax = np.linalg.norm(matched_pts1 - matched_pts2, axis=1)
    print(np.mean(parallax), np.std(parallax))
    az, pitch, roll = matcher.match(img1, img2, prev_frame_index=frame1)

    print("est azimuth:", az)
    print("est pitch  :", pitch)
    print("est roll   :", roll)
    gt_row = matcher.df_gt[matcher.df_gt["frame"] == frame2].iloc[0]
    gt_yaw = float(gt_row["yaw"])
    gt_pitch = float(gt_row["pitch"])
    gt_roll = float(gt_row["roll"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(img1, cmap="gray");
    ax[0].set_title(f"Frame {frame1}");
    ax[0].axis("off")
    ax[1].imshow(img2, cmap="gray");
    ax[1].set_title(f"Frame {frame2}");
    ax[1].axis("off")

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





if __name__ == '__main__':

    # run_simulator()
    run_RealPhoneCameraForward()
    # run_RealPhoneCameraRightToLeft()