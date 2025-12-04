# pose_from_two_images.py
from pathlib import Path
import numpy as np
import cv2

from image_loader import load_image_pair
from feature_extractor import create_feature_extractor, detect_and_compute
from matcher import create_matcher, match_descriptors

# כאן נניח שהפונקציות האלו קיימות אצלך כבר באיזה קובץ אחר
from gt_utils import load_gt_poses, evaluate_pair

def ypr_to_rotmat_y_up(roll, pitch, yaw):
    """
    Build rotation matrix R from:
        yaw   around +Y
        pitch around +X
        roll  around +Z

    Same convention used in rotmat_to_ypr_y_up:
        R = Ry(yaw) * Rx(pitch) * Rz(roll)
    """

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

    # final rotation matrix
    return Ry @ Rx @ Rz


def rotmat_to_ypr_y_up(R):
    """
    Decompose rotation matrix R to:
        yaw   around +Y
        pitch around +X
        roll  around +Z

    Assuming convention:
        R = Ry(yaw) * Rx(pitch) * Rz(roll)

    Returns: (roll, pitch, yaw) in radians
    """

    # מתוך הפיתוח הסימבולי:
    # R[1,2] = -sin(pitch)
    pitch = -np.arcsin(R[1, 2])

    # נזהרים מסינגולריות כשcos(pitch) ~ 0
    if np.isclose(np.cos(pitch), 0.0, atol=1e-6):
        # במקרה קצה – אפשר לקבע roll=0 ולחלץ yaw בערך מהאלמנטים היותר יציבים
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[0, 0])
    else:
        # R[0,2] = sin(yaw)*cos(pitch)
        # R[2,2] = cos(yaw)*cos(pitch)
        yaw = np.arctan2(R[0, 2], R[2, 2])

        # R[1,0] = sin(roll)*cos(pitch)
        # R[1,1] = cos(roll)*cos(pitch)
        roll = np.arctan2(R[1, 0], R[1, 1])

    return roll, pitch, yaw


def rad2deg(x):
    return x * 180.0 / np.pi


def rad2deg(x):
    return x * 180.0 / np.pi
def extract_matched_points(kp1, kp2, matches):
    """
    Turn cv2.DMatch + keypoints into Nx2 arrays of points.
    """
    pts1 = []
    pts2 = []
    for m in matches:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    return pts1, pts2


def compute_pose_from_images(
    img1,
    img2,
    K: np.ndarray,
    feature_method: str = "ORB",
    norm_type: str = "Hamming",
):
    """
    לוקח שתי תמונות + מטריצת כיול K ומחשב R,t ביניהן.
    """
    # 1) מאפיינים ותיאורים
    extractor = create_feature_extractor(feature_method)
    kp1, desc1 = detect_and_compute(img1, extractor)
    kp2, desc2 = detect_and_compute(img2, extractor)

    if desc1 is None or desc2 is None:
        raise RuntimeError("Could not compute descriptors for one of the images.")

    # 2) התאמנות
    matcher = create_matcher(norm_type=norm_type)
    matches = match_descriptors(
        desc1, desc2, matcher, sort_by_distance=True, max_matches=500
    )

    # 3) הפיכת ההתאמות לנקודות
    pts1, pts2 = extract_matched_points(kp1, kp2, matches)

    # 4) מטריצה אסנציאלית
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    if E is None:
        raise RuntimeError("Could not estimate Essential matrix.")

    # 5) שחזור R,t
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    return R, t, pts1, pts2, matches, mask_pose


if __name__ == "__main__":
    # ----------- שלב 1: טעינת שתי התמונות -----------
    base_dir = Path("../silmulator_data/simple_movement")

    frame1 = 0
    frame2 = 15
    # פה תעדכן את שמות הקבצים לפי מה שיש לך בפועל
    img1_path = base_dir / "images" / f"{str(frame1).zfill(6)}.png"
    img2_path = base_dir / "images" / f"{str(frame2).zfill(6)}.png"

    img1, img2 = load_image_pair(str(img1_path), str(img2_path), to_gray=True)

    # ----------- שלב 2: קליברציה לפי מה ששלחת -----------
    h, w = img1.shape[:2]
    scale_x = w / 960.0
    scale_y = h / 720.0
    fx = 924.82939686 * scale_x
    fy = 920.4766382 * scale_y
    cx = 468.24930789 * scale_x
    cy = 353.65863024 * scale_y

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])

    print("K =\n", K)

    # ----------- שלב 3: חישוב ה-Pose מהתמונות -----------
    R, t, pts1, pts2, matches, mask_pose = compute_pose_from_images(
        img1,
        img2,
        K,
        feature_method="ORB",
        norm_type="Hamming",
    )

    print("R =\n", R)
    print("t =\n", t)
    print("Number of inlier matches:", int(mask_pose.sum()))

    # ---- 4: GT מהקובץ ----
    poses_path = base_dir / "camera_poses.txt"
    df_gt = load_gt_poses(poses_path)



    # שורה של GT לכל פריים
    row1 = df_gt[df_gt["frame"] == frame1].iloc[0]
    row2 = df_gt[df_gt["frame"] == frame2].iloc[0]

    # זוויות GT גולמיות (נניח כבר בדגריז בקובץ)
    gt_roll1_deg = row1.roll
    gt_pitch1_deg = row1.pitch
    gt_yaw1_deg = row1.yaw

    gt_roll2_deg = row2.roll
    gt_pitch2_deg = row2.pitch
    gt_yaw2_deg = row2.yaw

    # דלתא זוויות GT (מה שכבר היה לך בעצם)
    gt_droll_deg = gt_roll2_deg - gt_roll1_deg
    gt_dpitch_deg = gt_pitch2_deg - gt_pitch1_deg
    gt_dyaw_deg = gt_yaw2_deg - gt_yaw1_deg

    # ---- 5: זוויות מוערכות מה-R של recoverPose ----
    roll_est, pitch_est, yaw_est = rotmat_to_ypr_y_up(R)

    est_roll_deg = rad2deg(roll_est)
    est_pitch_deg = rad2deg(pitch_est)
    est_yaw_deg = rad2deg(yaw_est)

    # ---- 6: הדפסה מסודרת להשוואה ----
    print(f"GT raw angles frame{frame1}: roll={gt_roll1_deg:.3f}, pitch={gt_pitch1_deg:.3f}, yaw={gt_yaw1_deg:.3f}")
    print(f"GT raw angles frame{frame2}: roll={gt_roll2_deg:.3f}, pitch={gt_pitch2_deg:.3f}, yaw={gt_yaw2_deg:.3f}")
    print(f"GT raw diff (deg):  Δroll={gt_droll_deg:.3f}, Δpitch={gt_dpitch_deg:.3f}, Δyaw={gt_dyaw_deg:.3f}")

    print(f"Estimated relative angles (deg): roll={est_roll_deg:.3f}, pitch={est_pitch_deg:.3f}, yaw={est_yaw_deg:.3f}")

    # אפשר גם שגיאה (אומדן - GT)
    err_roll = est_roll_deg - gt_droll_deg
    err_pitch = est_pitch_deg - gt_dpitch_deg
    err_yaw = est_yaw_deg - gt_dyaw_deg

    print(f"Angle errors (est - GT) (deg): dRoll={err_roll:.3f}, dPitch={err_pitch:.3f}, dYaw={err_yaw:.3f}")

    gt_x1, gt_y1, gt_z1 = row1.x, row1.y, row1.z
    gt_x2, gt_y2, gt_z2 = row2.x, row2.y, row2.z

    # --- Δpos ב-world frame כמו קודם ---
    gt_dx = gt_x2 - gt_x1
    gt_dy = gt_y2 - gt_y1
    gt_dz = gt_z2 - gt_z1
    gt_vec_world = np.array([gt_dx, gt_dy, gt_dz], dtype=float)
    gt_norm_world = np.linalg.norm(gt_vec_world)

    print(f"GT Δpos (world, X,Y,Z): dX={gt_dx:.6f}, dY={gt_dy:.6f}, dZ={gt_dz:.6f}")

    # --- בונים את R_wc1 (סיבוב מצלמה 1 ביחס לעולם) מ-GT roll/pitch/yaw ---
    roll1_rad = np.deg2rad(gt_roll1_deg)
    pitch1_rad = np.deg2rad(gt_pitch1_deg)
    yaw1_rad = np.deg2rad(gt_yaw1_deg)

    R_wc1 = ypr_to_rotmat_y_up(roll1_rad, pitch1_rad, yaw1_rad)  # world <- cam1
    R_c1w = R_wc1.T  # cam1 <- world

    # --- Δpos במערכת הצירים של המצלמה הראשונה ---
    gt_vec_cam1 = R_c1w @ gt_vec_world
    gt_norm_cam1 = np.linalg.norm(gt_vec_cam1)

    print(f"GT Δpos (cam1 frame): dX={gt_vec_cam1[0]:.6f}, dY={gt_vec_cam1[1]:.6f}, dZ={gt_vec_cam1[2]:.6f}")

    # --- t מה-recoverPose: וקטור תנועה במערכת הצירים של המצלמה הראשונה ---
    t_vec = t.reshape(3).astype(float)
    t_unit = t_vec / np.linalg.norm(t_vec)

    # נסדר סימן כך שיתיישר עם ה-GT במערכת הצירים של המצלמה
    if gt_norm_cam1 > 0:
        gt_unit_cam1 = gt_vec_cam1 / gt_norm_cam1
        dot = np.dot(t_unit, gt_unit_cam1)
        sign = np.sign(dot) if dot != 0 else 1.0
    else:
        sign = 1.0

    # עושים scale של t לפי האורך של GT במערכת הצירים של המצלמה
    t_scaled_cam1 = t_unit * gt_norm_cam1 * sign

    est_dx, est_dy, est_dz = t_scaled_cam1

    print(f"Estimated Δpos (cam1, scaled): dX={est_dx:.6f}, dY={est_dy:.6f}, dZ={est_dz:.6f}")

    err_dx = est_dx - gt_vec_cam1[0]
    err_dy = est_dy - gt_vec_cam1[1]
    err_dz = est_dz - gt_vec_cam1[2]

    print(f"Position errors in cam1 frame (est - GT): dX={err_dx:.6f}, dY={err_dy:.6f}, dZ={err_dz:.6f}")

    # שגיאה כוללת (נורמת L2 של וקטור השגיאה)
    err_vec = np.array([err_dx, err_dy, err_dz], dtype=float)
    err_norm = np.linalg.norm(err_vec)
    print(f"Total position error (cam1 frame, L2 norm): {err_norm:.6f} [m]")

