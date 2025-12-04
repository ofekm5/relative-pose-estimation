# gt_utils.py
import numpy as np
import pandas as pd
import re


def clean_header(name: str):
    """
    Clean a header:
    - remove spaces
    - remove parentheses
    - lower-case everything
    Example:
        ' pitch (deg) ' -> 'pitch'
        '   x  '       -> 'x'
    """
    name = name.strip()
    name = re.sub(r"\s+", "", name)      # remove all whitespace
    name = re.sub(r"\(.*?\)", "", name)  # remove parentheses
    return name.lower()


def load_gt_poses(path):
    """
    Load GT poses (camera_poses.txt).
    Expected real columns:
       frame, x, y, z, roll, pitch, yaw
    """

    df = pd.read_csv(path, sep=r"\s+", engine="python")

    # clean headers (just in case)
    df.columns = [clean_header(c) for c in df.columns]

    # keep only the correct fields
    expected = ["frame", "x", "y", "z", "roll", "pitch", "yaw"]
    df = df[expected]

    return df


def evaluate_pair(base_dir, df_gt, frame1: int, frame2: int):
    """
    מחשב שגיאות זווית ומיקום עבור זוג פריימים:
    מחזיר:
        pos_err  - נורמת השגיאה במיקום [m] במערכת הצירים של המצלמה הראשונה
        d_roll   - שגיאת זווית roll (deg)
        d_pitch  - שגיאת זווית pitch (deg)
        d_yaw    - שגיאת זווית yaw (deg)
    """
    import numpy as np
    from image_loader import load_image_pair
    # שואלים את pose_from_two_images על הפונקציות שעושות את העבודה
    from pose_from_two_images import (
        compute_pose_from_images,
        rotmat_to_ypr_y_up,
        ypr_to_rotmat_y_up,
        rad2deg,
    )

    # --- 1. טוענים תמונות ---
    img1_path = base_dir / "images" / f"{frame1:06d}.png"
    img2_path = base_dir / "images" / f"{frame2:06d}.png"
    img1, img2 = load_image_pair(str(img1_path), str(img2_path), to_gray=True)

    # --- 2. מטריצת כיול K (אותו חישוב כמו ב-main) ---
    h, w = img1.shape[:2]
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

    # --- 3. מחשבים R,t מהתמונות ---
    R, t, pts1, pts2, matches, mask_pose = compute_pose_from_images(
        img1, img2, K, feature_method="ORB", norm_type="Hamming"
    )

    # --- 4. GT מהטבלה ---
    row1 = df_gt[df_gt["frame"] == frame1].iloc[0]
    row2 = df_gt[df_gt["frame"] == frame2].iloc[0]

    gt_roll1_deg  = row1.roll
    gt_pitch1_deg = row1.pitch
    gt_yaw1_deg   = row1.yaw

    gt_roll2_deg  = row2.roll
    gt_pitch2_deg = row2.pitch
    gt_yaw2_deg   = row2.yaw

    # Δ זוויות GT
    gt_droll_deg  = gt_roll2_deg  - gt_roll1_deg
    gt_dpitch_deg = gt_pitch2_deg - gt_pitch1_deg
    gt_dyaw_deg   = gt_yaw2_deg   - gt_yaw1_deg

    # --- 5. זוויות מוערכות מ-R ---
    roll_est, pitch_est, yaw_est = rotmat_to_ypr_y_up(R)
    est_roll_deg  = rad2deg(roll_est)
    est_pitch_deg = rad2deg(pitch_est)
    est_yaw_deg   = rad2deg(yaw_est)

    # שגיאות זווית (אומדן - GT)
    d_roll  = est_roll_deg  - gt_droll_deg
    d_pitch = est_pitch_deg - gt_dpitch_deg
    d_yaw   = est_yaw_deg   - gt_dyaw_deg

    # --- 6. Δpos GT ב-world ---
    gt_vec_world = np.array([
        row2.x - row1.x,
        row2.y - row1.y,
        row2.z - row1.z
    ], dtype=float)

    # --- 7. מעבירים את Δpos למערכת הצירים של המצלמה הראשונה ---
    roll1_rad  = np.deg2rad(gt_roll1_deg)
    pitch1_rad = np.deg2rad(gt_pitch1_deg)
    yaw1_rad   = np.deg2rad(gt_yaw1_deg)

    # world <- cam1
    R_wc1 = ypr_to_rotmat_y_up(roll1_rad, pitch1_rad, yaw1_rad)
    # cam1 <- world
    R_c1w = R_wc1.T

    gt_vec_cam1  = R_c1w @ gt_vec_world
    gt_norm_cam1 = np.linalg.norm(gt_vec_cam1)

    # --- 8. t מ-recoverPose: נורמליזציה + SCALE לפי |GT| ---
    t_vec  = t.reshape(3).astype(float)
    t_norm = np.linalg.norm(t_vec)

    if t_norm < 1e-9 or gt_norm_cam1 < 1e-9:
        # אין מידע טוב → נחזיר NaN
        pos_err = float("nan")
        return pos_err, float(d_roll), float(d_pitch), float(d_yaw)

    t_unit       = t_vec / t_norm
    gt_unit_cam1 = gt_vec_cam1 / gt_norm_cam1

    # מסדרים סימן לפי GT
    dot  = np.dot(t_unit, gt_unit_cam1)
    sign = np.sign(dot) if dot != 0 else 1.0

    t_scaled_cam1 = t_unit * gt_norm_cam1 * sign

    # --- 9. שגיאת מיקום אמיתית (cam1 frame) ---
    err_vec = t_scaled_cam1 - gt_vec_cam1
    pos_err = float(np.linalg.norm(err_vec))

    return pos_err, float(d_roll), float(d_pitch), float(d_yaw)
