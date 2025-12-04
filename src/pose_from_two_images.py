# pose_from_two_images.py
from pathlib import Path
import numpy as np
import cv2

from image_loader import load_image_pair
from feature_extractor import create_feature_extractor, detect_and_compute
from matcher import create_matcher, match_descriptors

# כאן נניח שהפונקציות האלו קיימות אצלך כבר באיזה קובץ אחר
from gt_utils import load_gt_poses, evaluate_pair



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

    # פה תעדכן את שמות הקבצים לפי מה שיש לך בפועל
    img1_path = base_dir / "images" / "000330.png"
    img2_path = base_dir / "images" / "000345.png"

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

    frame1 = 0
    frame2 = 15

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
    est_dx, est_dy, est_dz = t[0,0], t[1,0], t[2,0]

    gt_dx = gt_x2 - gt_x1
    gt_dy = gt_y2 - gt_y1
    gt_dz = gt_z2 - gt_z1
    print(f"GT Δpos (X,Y,Z): dX={gt_dx:.6f}, dY={gt_dy:.6f}, dZ={gt_dz:.6f}")
    err_dx = est_dx - gt_dx
    err_dy = est_dy - gt_dy
    err_dz = est_dz - gt_dz
    print(f"Position errors (est - GT): dX={err_dx:.6f}, dY={err_dy:.6f}, dZ={err_dz:.6f}")
