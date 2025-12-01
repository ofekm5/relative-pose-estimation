import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pose_from_two_images import (
    detect_orb,
    match_descriptors,
    rotation_matrix_to_euler_angles,
    rad2deg,
)

# ================= CAMERA PARAMETERS (מהסימולטור) =================
CAMERA_FX = 924.82939686
CAMERA_FY = 920.4766382
CAMERA_CX = 468.24930789
CAMERA_CY = 353.65863024

CAMERA_K1 =  2.34795296e-02
CAMERA_K2 = -4.38293524e-01
CAMERA_P1 = -1.48496922e-03
CAMERA_P2 = -8.11809445e-06

CAMERA_WIDTH  = 960
CAMERA_HEIGHT = 720

FRAME_STEP = 10
MAX_FRAMES = 400

# ================= PATHS =================
BASE_DIR = "../silmulator_data"   # TODO: לשנות לנתיב שממנו C++ שמר
IMAGES_DIR = os.path.join(BASE_DIR, "images")
POSES_PATH = os.path.join(BASE_DIR, "camera_poses.csv")


# ============================================================
# 1. טעינת GT
# ============================================================

def load_poses():
    df = pd.read_csv(POSES_PATH)
    df = df.sort_values("frame").reset_index(drop=True)
    return df


def build_intrinsics(width, height):
    # קליברציה מקורית
    W_ORIG = CAMERA_WIDTH   # 960
    H_ORIG = CAMERA_HEIGHT  # 720

    scale_x = width / W_ORIG
    scale_y = height / H_ORIG

    fx = CAMERA_FX * scale_x
    fy = CAMERA_FY * scale_y
    cx = CAMERA_CX * scale_x
    cy = CAMERA_CY * scale_y

    if (abs(scale_x - 1.0) > 1e-6) or (abs(scale_y - 1.0) > 1e-6):
        print(f"[INFO] scaling intrinsics: "
              f"orig=({W_ORIG}x{H_ORIG}), new=({width}x{height}), "
              f"scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return K


# ============================================================
# 2. המרת RPY של ה-GT למטריצת רוטציה (Rwc)
#    שימוש באותה קונבנציה כמו בסימולטור:
#    Rwc = Rz(yaw) * Ry(pitch) * Rx(roll)
# ============================================================

def euler_rpy_deg_to_Rwc(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rz = np.array([
        [ cy, -sy, 0.0],
        [ sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ])
    Ry = np.array([
        [ cp, 0.0,  sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0,  cp],
    ])
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  cr, -sr],
        [0.0,  sr,  cr],
    ])

    Rwc = Rz @ Ry @ Rx
    return Rwc


# ============================================================
# 3. חישוב פוז יחסית מ-frame 0 לכל frame k
# ============================================================

def estimate_pose_0_to_k(img0_gray, kp0, des0, imgk_gray, K):
    """
    מחזיר:
        success (bool),
        Ck (3,) - מיקום מצלמה k במערכת הצירים של frame 0 (up to scale),
        R_0k (3x3) - רוטציה מ-0 ל-k,
        roll, pitch, yaw (deg) עבור רוטציה 0->k (לפי rotation_matrix_to_euler_angles שלך)
    """
    kpK, desK = detect_orb(imgk_gray, nfeatures=3000)

    if des0 is None or desK is None or len(kp0) < 10 or len(kpK) < 10:
        print("[WARN] not enough keypoints/descriptors (0->k)")
        return False, None, None, None, None, None

    good_matches = match_descriptors(des0, desK, ratio_thresh=0.75)
    print(f"Good matches (0 -> k): {len(good_matches)}")
    if len(good_matches) < 8:
        print("[WARN] not enough matches to estimate pose (0->k)")
        return False, None, None, None, None, None

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good_matches])
    ptsK = np.float32([kpK[m.trainIdx].pt for m in good_matches])

    E, mask_E = cv2.findEssentialMat(
        pts0,
        ptsK,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    if E is None:
        print("[WARN] cannot estimate Essential matrix (0->k)")
        return False, None, None, None, None, None

    try:
        _, R, t, mask_pose = cv2.recoverPose(E, pts0, ptsK, K)
    except cv2.error as e:
        print("[WARN] recoverPose failed (0->k):", e)
        return False, None, None, None, None, None

    # מיקום מצלמה k במערכת 0: Ck = -R^T t
    Ck = -R.T @ t
    Ck = Ck.flatten()

    # orientation (אותו כיוון כמו בפונקציה שלך)
    yaw, pitch, roll = rotation_matrix_to_euler_angles(R)
    yaw_deg = rad2deg(yaw)
    pitch_deg = rad2deg(pitch)
    roll_deg = rad2deg(roll)

    return True, Ck, R, roll_deg, pitch_deg, yaw_deg


# ============================================================
# 4. Umeyama / Similarity alignment: GT <-> EST (translation)
# ============================================================

def align_similarity(est_points, gt_points):
    """
    est_points: (N,3) - מהאלגוריתם, במערכת צירים של frame 0, בלי סקאלה
    gt_points:  (N,3) - GT מהסימולטור, במערכת העולמית

    מחזיר:
        s, R, t  כך ש:
        p_gt ≈ s * R @ p_est + t
    (רוטציה זו שייכת להתאמה של המסלול התרגומי; לא רוטציית המצלמה עצמה)
    """
    est = np.asarray(est_points, dtype=np.float64)
    gt  = np.asarray(gt_points,  dtype=np.float64)

    assert est.shape == gt.shape
    N = est.shape[0]

    mu_est = est.mean(axis=0)
    mu_gt  = gt.mean(axis=0)

    est_c = est - mu_est
    gt_c  = gt  - mu_gt

    H = est_c.T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # טיפול בהיפוך ציר
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    var_est = np.sum(est_c ** 2)
    if var_est < 1e-12:
        s = 1.0
    else:
        s = np.sum(S) / var_est

    mu_est_col = mu_est.reshape(3, 1)
    mu_gt_col  = mu_gt.reshape(3, 1)
    t_vec = mu_gt_col - s * R @ mu_est_col
    t_vec = t_vec.flatten()

    return s, R, t_vec


def apply_similarity(est_points, s, R, t_vec):
    est = np.asarray(est_points, dtype=np.float64)
    est_aligned = s * (est @ R.T)
    est_aligned += t_vec
    return est_aligned


# ============================================================
# 5. main: בניית מסלול, alignment, שגיאות ו־viewer
# ============================================================

def main():
    poses_df = load_poses()
    n_rows = len(poses_df)
    print(f"Loaded {n_rows} GT rows from {POSES_PATH}")

    if n_rows < 2:
        print("Not enough frames.")
        return

    # Frame 0 כרפרנס
    row0 = poses_df.iloc[0]
    frame0 = int(row0["frame"])
    roll0_gt = float(row0["roll"])
    pitch0_gt = float(row0["pitch"])
    yaw0_gt = float(row0["yaw"])

    Rwc_0 = euler_rpy_deg_to_Rwc(roll0_gt, pitch0_gt, yaw0_gt)

    img0_path = os.path.join(IMAGES_DIR, f"{frame0:06d}.png")
    img0_gray = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    if img0_gray is None:
        print(f"[ERROR] cannot load reference image: {img0_path}")
        return

    h0, w0 = img0_gray.shape[:2]
    K = build_intrinsics(w0, h0)
    print("[INFO] computing ORB on frame 0 once...")
    kp0, des0 = detect_orb(img0_gray, nfeatures=3000)
    if des0 is None or len(kp0) < 10:
        print("[ERROR] not enough keypoints in frame 0")
        return

    est_points = []   # תרגום מהאלגוריתם (up to scale)
    gt_points  = []   # GT מוחלטות
    est_Rs     = []   # רוטציה 0->k מהאלגוריתם
    gt_Rs      = []   # רוטציה 0->k מה-GT
    frames_used = []
    est_rpys  = []    # (roll_est, pitch_est, yaw_est) ב־degrees
    gt_rpys   = []    # (roll_gt_rel, pitch_gt_rel, yaw_gt_rel) ב־degrees

    indices = list(range(1, n_rows, FRAME_STEP))
    if len(indices) > MAX_FRAMES:
        indices = indices[:MAX_FRAMES]

    print(f"[INFO] using {len(indices)} frames out of {n_rows}")
    # נעבור על כל הפריימים ונחשב pose 0->k
    for idx in indices:
        rowk = poses_df.iloc[idx]
        fk = int(rowk["frame"])

        imgk_path = os.path.join(IMAGES_DIR, f"{fk:06d}.png")
        imgk_gray = cv2.imread(imgk_path, cv2.IMREAD_GRAYSCALE)
        if imgk_gray is None:
            print(f"[WARN] cannot load image for frame {fk}: {imgk_path}")
            continue

        ok, Ck, R_0k_est, roll_est, pitch_est, yaw_est = estimate_pose_0_to_k(
            img0_gray, kp0, des0, imgk_gray, K
        )
        if not ok:
            print(f"[INFO] skipping frame {fk} due to pose failure")
            continue

        ok, Ck, R_0k_est, roll_est, pitch_est, yaw_est = estimate_pose_0_to_k(
            img0_gray, kp0, des0, imgk_gray, K
        )
        if not ok:
            print(f"[INFO] skipping frame {fk} due to pose failure")
            continue

        # GT position of frame k
        xk, yk, zk = float(rowk["x"]), float(rowk["y"]), float(rowk["z"])
        rollk_gt = float(rowk["roll"])
        pitchk_gt = float(rowk["pitch"])
        yawk_gt = float(rowk["yaw"])

        # רוטציה מוחלטת של frame k
        Rwc_k = euler_rpy_deg_to_Rwc(rollk_gt, pitchk_gt, yawk_gt)

        # רוטציה יחסית 0->k ב-GT
        R_0k_GT = Rwc_k.T @ Rwc_0

        # -------------------------------------------------
        # כאן אנחנו מפיקים גם RPY יחסיים ל-GT (0->k)
        # -------------------------------------------------
        yaw_gt_rel, pitch_gt_rel, roll_gt_rel = rotation_matrix_to_euler_angles(R_0k_GT)
        roll_gt_rel_deg  = rad2deg(roll_gt_rel)
        pitch_gt_rel_deg = rad2deg(pitch_gt_rel)
        yaw_gt_rel_deg   = rad2deg(yaw_gt_rel)

        # נשמור את הכל ברשימות
        est_points.append(Ck)
        gt_points.append([xk, yk, zk])
        est_Rs.append(R_0k_est)
        gt_Rs.append(R_0k_GT)
        est_rpys.append((roll_est, pitch_est, yaw_est))  # כבר ב-deg
        gt_rpys.append((roll_gt_rel_deg, pitch_gt_rel_deg, yaw_gt_rel_deg))
        frames_used.append(fk)

    est_points = np.array(est_points)
    gt_points  = np.array(gt_points)
    est_Rs     = np.array(est_Rs)
    gt_Rs      = np.array(gt_Rs)
    frames_used = np.array(frames_used, dtype=int)
    est_rpys = np.array(est_rpys)  # shape (N, 3)
    gt_rpys  = np.array(gt_rpys)   # shape (N, 3)

    print(f"Used {len(frames_used)} frames for alignment.")

    if len(frames_used) < 3:
        print("Not enough frames for stable alignment.")
        return

    # ---- alignment תרגומי (Umeyama) ----
    s, R_align, t_align = align_similarity(est_points, gt_points)
    print("Alignment parameters (translation track):")
    print("  scale s =", s)
    print("  R_align =\n", R_align)
    print("  t_align =", t_align)

    est_aligned = apply_similarity(est_points, s, R_align, t_align)

    # ====================================================
    # חישוב שגיאות תרגום ורוטציה
    # ====================================================
    transl_err_norms = []
    rot_err_degs = []

    for i in range(len(frames_used)):
        gt_p = gt_points[i]
        est_p = est_aligned[i]
        err = est_p - gt_p
        transl_err_norms.append(np.linalg.norm(err))

        R_est = est_Rs[i]
        R_gt  = gt_Rs[i]

        # R_err = R_est^T * R_gt
        R_err = R_est.T @ R_gt
        trace_val = np.trace(R_err)
        # הגנה על נומריקה
        val = max(min((trace_val - 1.0) / 2.0, 1.0), -1.0)
        angle_err_rad = math.acos(val)
        angle_err_deg = math.degrees(angle_err_rad)
        rot_err_degs.append(angle_err_deg)

    transl_err_norms = np.array(transl_err_norms)
    rot_err_degs = np.array(rot_err_degs)

    rmse_trans = math.sqrt(np.mean(transl_err_norms ** 2))
    mean_rot_err = np.mean(rot_err_degs)

    print(f"Translation RMSE = {rmse_trans:.4f}")
    print(f"Mean rotation error (deg) = {mean_rot_err:.4f}")

    # ====================================================
    # Viewer: לכל פריים – תמונה + GT + EST + err
    # ====================================================
    plt.ion()
    fig_img, ax_img = plt.subplots(figsize=(8, 5))

    for i in range(len(frames_used)):
        fk = frames_used[i]

        img_path = os.path.join(IMAGES_DIR, f"{fk:06d}.png")
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gt_p = gt_points[i]
        est_p = est_aligned[i]
        err_vec = est_p - gt_p
        err_norm = transl_err_norms[i]
        rot_err = rot_err_degs[i]

        # 6 DOF GT ו-EST (יחסיים בין frame0 ל-framek)
        roll_gt, pitch_gt, yaw_gt = gt_rpys[i]    # ב-deg
        roll_est, pitch_est, yaw_est = est_rpys[i]

        ax_img.clear()
        ax_img.imshow(img_rgb)
        ax_img.axis("off")
        ax_img.set_title(f"Frame {fk}: GT vs EST (aligned)", fontsize=12)

        txt = (
            f"GT pos:    {gt_p[0]: .3f}  {gt_p[1]: .3f}  {gt_p[2]: .3f}\n"
            f"GT rpy:    roll={roll_gt: .2f}  pitch={pitch_gt: .2f}  yaw={yaw_gt: .2f}\n"
            f"EST pos:   {est_p[0]: .3f}  {est_p[1]: .3f}  {est_p[2]: .3f}\n"
            f"EST rpy:   roll={roll_est: .2f}  pitch={pitch_est: .2f}  yaw={yaw_est: .2f}\n"
            f"ERR pos:   {err_vec[0]: .3f}  {err_vec[1]: .3f}  {err_vec[2]: .3f}   |err|={err_norm:.3f}\n"
            f"Rot err:   {rot_err:.3f} deg"
        )

        ax_img.text(
            0.02,
            0.95,
            txt,
            color="yellow",
            fontsize=10,
            transform=ax_img.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        )

        plt.draw()
        plt.pause(0.3)


    print("Done showing frames – close window to continue.")
    plt.ioff()
    plt.show()

    # ====================================================
    # גרפים: שגיאת תרגום ורוטציה לאורך הזמן
    # ====================================================
    fig_err, (ax_t, ax_r) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_t.plot(frames_used, transl_err_norms, marker='o')
    ax_t.set_ylabel("||translation error|| [world units]")
    ax_t.grid(True)
    ax_t.set_title(f"Translation error (RMSE={rmse_trans:.3f})")

    ax_r.plot(frames_used, rot_err_degs, marker='o')
    ax_r.set_ylabel("rotation error [deg]")
    ax_r.set_xlabel("frame index")
    ax_r.grid(True)
    ax_r.set_title(f"Rotation error (mean={mean_rot_err:.3f} deg)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
