import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import itertools


class TwoViewPose:
    # ------------------------------
    # Intrinsics (K)
    # ------------------------------
    @staticmethod
    def _build_K(image):
        h, w = image.shape[:2]
        scale_x = w / 960.0
        scale_y = h / 720.0

        fx = 924.82939686 * scale_x
        fy = 920.4766382 * scale_y
        cx = 468.24930789 * scale_x
        cy = 353.65863024 * scale_y

        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0,  0,  1]], dtype=np.float64)

    @staticmethod
    def _read_gray(path: str) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img

    # ------------------------------
    # LSD lines
    # ------------------------------
    @staticmethod
    def _detect_lsd_lines(gray: np.ndarray):
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines, _, _, _ = lsd.detect(gray)
        if lines is None:
            return np.zeros((0, 4), dtype=np.float64)
        return lines.reshape(-1, 4).astype(np.float64)

    @staticmethod
    def _draw_lines(gray: np.ndarray, lines: np.ndarray, color=(0, 255, 0)):
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in lines:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, cv2.LINE_AA)
        return vis

    # ------------------------------
    # Your YPR convention
    # yaw around +Y, pitch around +X, roll around +Z
    # ------------------------------
    @staticmethod
    def _ypr_to_rotmat_y_up(roll_deg, pitch_deg, yaw_deg):
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)

        cy, sy = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[ cy, 0.0, sy],
                       [0.0, 1.0, 0.0],
                       [-sy, 0.0, cy]], dtype=np.float64)

        cx, sx = np.cos(pitch), np.sin(pitch)
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0,  cx, -sx],
                       [0.0,  sx,  cx]], dtype=np.float64)

        cz, sz = np.cos(roll), np.sin(roll)
        Rz = np.array([[ cz, -sz, 0.0],
                       [ sz,  cz, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)

        return Ry @ Rx @ Rz

    @staticmethod
    def rotmat_to_ypr_y_up(R):
        pitch = np.arcsin(R[2, 1])
        if abs(R[2, 1]) > 0.9999:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            yaw = 0.0
            return roll, pitch, yaw
        yaw = np.arctan2(-R[2, 0], R[0, 0])
        roll = np.arctan2(R[1, 0], R[1, 1])
        return roll, pitch, yaw

    # ------------------------------
    # Error utils
    # ------------------------------
    @staticmethod
    def wrap_deg(a):
        return (a + 180.0) % 360.0 - 180.0

    @staticmethod
    def rotation_angle_error_deg(R_est, R_gt):
        R_err = R_gt.T @ R_est
        val = (np.trace(R_err) - 1.0) / 2.0
        val = np.clip(val, -1.0, 1.0)
        return np.rad2deg(np.arccos(val))

    # ------------------------------
    # ORB -> Essential -> recoverPose
    # ------------------------------
    @staticmethod
    def estimate_pose_points(img1, img2, K, nfeatures=4000, ratio=0.75, ransac_prob=0.999, ransac_thresh_px=1.0):
        orb = cv2.ORB_create(nfeatures=nfeatures)
        k1, d1 = orb.detectAndCompute(img1, None)
        k2, d2 = orb.detectAndCompute(img2, None)

        if d1 is None or d2 is None or k1 is None or k2 is None or len(k1) < 8 or len(k2) < 8:
            return None, None, {"ok": False, "reason": "Not enough keypoints/descriptors"}, {}

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(d1, d2, k=2)

        good = []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)

        debug = {}
        debug["matches_good"] = cv2.drawMatches(
            img1, k1, img2, k2, good, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        if len(good) < 8:
            return None, None, {"ok": False, "reason": "Not enough good matches"}, debug

        pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        pts2 = np.float32([k2[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=ransac_prob, threshold=ransac_thresh_px)
        if E is None or mask is None:
            return None, None, {"ok": False, "reason": "findEssentialMat failed"}, debug

        inlier_mask = mask.ravel().astype(bool)
        inlier_matches = [m for m, keep in zip(good, inlier_mask) if keep]

        debug["matches_inliers_E"] = cv2.drawMatches(
            img1, k1, img2, k2, inlier_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        pts1_in = pts1[inlier_mask]
        pts2_in = pts2[inlier_mask]
        if len(pts1_in) < 5:
            return None, None, {"ok": False, "reason": "Not enough inliers for recoverPose"}, debug

        inliers_count, R_rel, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)

        stats = {
            "ok": True,
            "good_matches": len(good),
            "inliers_E": int(inlier_mask.sum()),
            "recoverPose_inliers": int(inliers_count),
        }
        return R_rel, t, stats, debug

    # ============================================================
    # VP extraction (polar grid voting + weight |l0||l1|sin(2θ))
    # ============================================================
    @staticmethod
    def _line_angle_and_len(line):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy) + 1e-9
        angle = np.arctan2(dy, dx)
        return angle, length

    @staticmethod
    def _hom_line_from_segment(line):
        x1, y1, x2, y2 = line
        p1 = np.array([x1, y1, 1.0])
        p2 = np.array([x2, y2, 1.0])
        l = np.cross(p1, p2)
        n = np.linalg.norm(l[:2]) + 1e-12
        return l / n

    @staticmethod
    def _vp_dir_from_image_point(vp_xy, K):
        x, y = vp_xy
        v = np.array([x, y, 1.0], dtype=np.float64)
        d = np.linalg.inv(K) @ v
        d = d / (np.linalg.norm(d) + 1e-12)
        if d[2] < 0:
            d = -d
        return d

    @staticmethod
    def _dir_to_grid_idx(d, n_lat=90, n_lon=360):
        lat = np.arctan2(np.hypot(d[0], d[1]), d[2])  # 0..pi/2
        lon = np.arctan2(d[1], d[0])                  # -pi..pi
        lat_deg = np.rad2deg(lat)
        lon_deg = (np.rad2deg(lon) + 360.0) % 360.0

        lat_i = int(np.clip(lat_deg, 0, n_lat - 1))
        lon_i = int(np.clip(lon_deg, 0, n_lon - 1))
        return lat_i, lon_i

    @staticmethod
    def estimate_manhattan_dirs_from_image(gray, K,
                                           max_lines=200,
                                           n_lat=90,
                                           n_lon=360,
                                           max_pairs=15000,
                                           rng_seed=0):
        """
        Returns:
          Delta (3x3) columns = δ_k (camera-frame Manhattan directions)
          ok
          dbg: num_lines, acc_max, lines_used, vp2_score
        """
        lines = TwoViewPose._detect_lsd_lines(gray)
        dbg = {"num_lines": int(lines.shape[0])}
        if lines.shape[0] < 10:
            return None, False, dbg

        lens_all = np.array([TwoViewPose._line_angle_and_len(l)[1] for l in lines])
        idx = np.argsort(-lens_all)[:min(max_lines, len(lines))]
        lines = lines[idx]
        lens = lens_all[idx]

        hlines = np.array([TwoViewPose._hom_line_from_segment(l) for l in lines])
        angles = np.array([TwoViewPose._line_angle_and_len(l)[0] for l in lines])

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

            w = lens[i] * lens[j] * abs(np.sin(2.0 * theta))
            if w <= 0:
                continue

            d = TwoViewPose._vp_dir_from_image_point(vp_xy, K)
            lat_i, lon_i = TwoViewPose._dir_to_grid_idx(d, n_lat=n_lat, n_lon=n_lon)
            acc[lat_i, lon_i] += w

        acc_max = float(np.max(acc))
        dbg["acc_max"] = acc_max
        dbg["lines_used"] = int(m)

        if acc_max <= 0:
            return None, False, dbg

        # VP1
        lat1, lon1 = np.unravel_index(np.argmax(acc), acc.shape)
        lat1_rad = np.deg2rad(lat1 + 0.5)
        lon1_rad = np.deg2rad(lon1 + 0.5)
        v1 = np.array([
            np.sin(lat1_rad) * np.cos(lon1_rad),
            np.sin(lat1_rad) * np.sin(lon1_rad),
            np.cos(lat1_rad)
        ], dtype=np.float64)
        v1 /= (np.linalg.norm(v1) + 1e-12)

        # VP2 on great circle ⟂ v1
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, v1)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        a = np.cross(v1, tmp); a /= (np.linalg.norm(a) + 1e-12)
        b = np.cross(v1, a);   b /= (np.linalg.norm(b) + 1e-12)

        best_score = -1.0
        v2 = None
        for deg in range(360):
            ang = np.deg2rad(deg)
            cand = np.cos(ang) * a + np.sin(ang) * b
            cand /= (np.linalg.norm(cand) + 1e-12)
            lat_i, lon_i = TwoViewPose._dir_to_grid_idx(cand, n_lat=n_lat, n_lon=n_lon)
            s = acc[lat_i, lon_i]
            if s > best_score:
                best_score = s
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
    # Paper-style Absolute Rotation Optimization using VP dirs
    # E(R) = sum_k arccos( δ_k · (R d_k) )
    # ============================================================
    @staticmethod
    def so3_exp(w):
        R, _ = cv2.Rodrigues(w.reshape(3, 1))
        return R

    @staticmethod
    def vp_cost(R_iw, Delta_cam, D_world):
        # sum_k arccos( δ · (R d) )
        cost = 0.0
        for k in range(3):
            delta = Delta_cam[:, k]
            d = D_world[:, k]
            u = R_iw @ d
            s = float(np.clip(delta @ u, -1.0, 1.0))
            cost += float(np.arccos(s))
        return cost

    @staticmethod
    def optimize_absolute_rotation_from_vps(R_init, Delta_cam, D_world, iters=12, lm_lambda=1e-2):
        R = R_init.copy()

        for _ in range(iters):
            r_list = []
            J_list = []

            for k in range(3):
                delta = Delta_cam[:, k]
                d = D_world[:, k]
                u = R @ d

                s = float(np.clip(delta @ u, -1.0, 1.0))
                e = np.arccos(s)
                r_list.append(e)

                denom = np.sqrt(max(1e-12, 1.0 - s * s))
                cross = np.cross(delta, u)     # (3,)
                J = -(1.0 / denom) * cross    # (3,)
                J_list.append(J.reshape(1, 3))

            r = np.array(r_list).reshape(3, 1)   # (3,1)
            J = np.vstack(J_list)                # (3,3)

            H = J.T @ J + lm_lambda * np.eye(3)
            g = J.T @ r
            try:
                dw = -np.linalg.solve(H, g).reshape(3,)
            except np.linalg.LinAlgError:
                break

            R = TwoViewPose.so3_exp(dw) @ R
            if np.linalg.norm(dw) < 1e-7:
                break

        return R


def main():
    frame1 = 230
    frame2 = 245
    base_dir = "/home/orr/university_projects/relative-pose-estimation/silmulator_data/simple_movement"

    # =========================
    # VP settings (tune here)
    # =========================
    use_vp_refinement = True

    vp_max_lines = 200
    vp_max_pairs = 15000

    # Reliability gates (מה שפתר לך את המקרה ש-new חלש)
    ACC_MIN = 8e5
    VP2_MIN = 8000

    # Extra acceptance gate (בלי GT): accept only if VP cost improves
    COST_IMPROVE_EPS = 1e-3  # radians

    img1_path = Path(base_dir) / "images" / f"{frame1:06d}.png"
    img2_path = Path(base_dir) / "images" / f"{frame2:06d}.png"

    img1 = TwoViewPose._read_gray(img1_path)
    img2 = TwoViewPose._read_gray(img2_path)
    K = TwoViewPose._build_K(img1)

    print("K=\n", K)

    # 1) point-based relative pose
    R_rel_pts, t, stats, debug = TwoViewPose.estimate_pose_points(img1, img2, K)
    if not stats.get("ok", False):
        print("Pose failed:", stats.get("reason"))
        return

    print("\n=== Points (ORB/E) ===")
    print("R_rel_pts=\n", R_rel_pts)
    print("t=\n", t)

    # 2) GT rotations (for evaluation + building D_world from prev frame)
    df_gt = pd.read_csv(Path(base_dir) / "camera_poses.txt", sep=r"\s+")
    prev_row = df_gt[df_gt["frame"] == frame1].iloc[0]
    row2 = df_gt[df_gt["frame"] == frame2].iloc[0]

    R_prev_world = TwoViewPose._ypr_to_rotmat_y_up(float(prev_row["roll"]),
                                                  float(prev_row["pitch"]),
                                                  float(prev_row["yaw"]))
    R_gt2_world = TwoViewPose._ypr_to_rotmat_y_up(float(row2["roll"]),
                                                 float(row2["pitch"]),
                                                 float(row2["yaw"]))

    # ✅ Correct composition (Option B)
    R_new_init = R_prev_world @ R_rel_pts
    R_new_world = R_new_init.copy()

    # 3) VP extraction
    Delta_prev, ok1, dbg1 = TwoViewPose.estimate_manhattan_dirs_from_image(
        img1, K, max_lines=vp_max_lines, max_pairs=vp_max_pairs, rng_seed=0
    )
    Delta_new, ok2, dbg2 = TwoViewPose.estimate_manhattan_dirs_from_image(
        img2, K, max_lines=vp_max_lines, max_pairs=vp_max_pairs, rng_seed=1
    )

    print("\n=== VP debug ===")
    print("prev:", dbg1)
    print("new :", dbg2)

    if use_vp_refinement and ok1 and ok2:
        vp_good_prev = (dbg1.get("acc_max", 0.0) >= ACC_MIN) and (dbg1.get("vp2_score", 0.0) >= VP2_MIN)
        vp_good_new  = (dbg2.get("acc_max", 0.0) >= ACC_MIN) and (dbg2.get("vp2_score", 0.0) >= VP2_MIN)

        if vp_good_prev and vp_good_new:
            # From δ ~ R d  => d = R^T δ  (using prev frame)
            D_world = R_prev_world.T @ Delta_prev

            cost_init = TwoViewPose.vp_cost(R_new_init, Delta_new, D_world)

            R_opt = TwoViewPose.optimize_absolute_rotation_from_vps(
                R_init=R_new_init,
                Delta_cam=Delta_new,
                D_world=D_world,
                iters=12,
                lm_lambda=1e-2
            )

            cost_opt = TwoViewPose.vp_cost(R_opt, Delta_new, D_world)

            print(f"\n[VP] cost_init={cost_init:.6f} rad, cost_opt={cost_opt:.6f} rad")

            # accept only if VP explains the observations better
            if cost_opt < cost_init - COST_IMPROVE_EPS:
                R_new_world = R_opt
                print("[OK] VP refinement accepted (passed reliability + cost gate).")
            else:
                R_new_world = R_new_init
                print("[WARN] VP refinement rejected (no cost improvement) -> points-only.")
        else:
            R_new_world = R_new_init
            print("[WARN] VP skipped (weak/unstable by acc/vp2 gates) -> points-only.")
    else:
        R_new_world = R_new_init
        print("\n[INFO] VP refinement disabled or VP extraction failed -> points-only.")

    # 4) back to YPR
    roll_est, pitch_est, yaw_est = TwoViewPose.rotmat_to_ypr_y_up(R_new_world)
    est_roll = np.rad2deg(roll_est)
    est_pitch = np.rad2deg(pitch_est)
    est_yaw = np.rad2deg(yaw_est)

    gt_yaw = float(row2["yaw"])
    gt_pitch = float(row2["pitch"])
    gt_roll = float(row2["roll"])

    yaw_err = TwoViewPose.wrap_deg(est_yaw - gt_yaw)
    rot_err = TwoViewPose.rotation_angle_error_deg(R_new_world, R_gt2_world)

    print("\n=== EST world angles (frame2) ===")
    print(f"EST yaw  = {est_yaw:.6f}")
    print(f"EST pitch= {est_pitch:.6f}")
    print(f"EST roll = {est_roll:.6f}")

    print("\n=== GT angles (frame2) ===")
    print(f"GT yaw   = {gt_yaw:.6f}")
    print(f"GT pitch = {gt_pitch:.6f}")
    print(f"GT roll  = {gt_roll:.6f}")

    print("\n=== Errors ===")
    print(f"Yaw error wrapped (deg)   = {yaw_err:.6f}")
    print(f"Rotation angle error (deg)= {rot_err:.6f}")

    # Visuals
    lines1 = TwoViewPose._detect_lsd_lines(img1)
    lines2 = TwoViewPose._detect_lsd_lines(img2)
    vis1 = TwoViewPose._draw_lines(img1, lines1[:200], color=(0, 255, 0))
    vis2 = TwoViewPose._draw_lines(img2, lines2[:200], color=(0, 255, 0))

    cv2.imshow("LSD lines - img1", vis1)
    cv2.imshow("LSD lines - img2", vis2)
    if "matches_good" in debug:
        cv2.imshow("ORB good matches", debug["matches_good"])
    if "matches_inliers_E" in debug:
        cv2.imshow("Inlier matches (E RANSAC)", debug["matches_inliers_E"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
