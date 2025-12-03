import cv2
import numpy as np
import math
import pandas as pd
from pathlib import Path


# ==========================
# 1. CONFIGURATION
# ==========================

# Base directory where your simulator output is stored:
#   BASE_DIR/
#     images/000000.png, 000001.png, ...
#     camera_poses.csv
BASE_DIR = Path("../silmulator_data")   # TODO: change if needed

IMAGES_DIR = BASE_DIR / "images"
POSES_CSV = BASE_DIR / "camera_poses.csv"

# Choose which frames to compare
FRAME1 = 0
FRAME2 = 30   # for example; change to any pair you want


# Camera calibration (original resolution: 960x720)
CAMERA_FX = 924.82939686
CAMERA_FY = 920.4766382
CAMERA_CX = 468.24930789
CAMERA_CY = 353.65863024
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720


# ==========================
# 2. UTILS
# ==========================

def load_gray_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def build_intrinsics(width: int, height: int) -> np.ndarray:
    """
    Build camera matrix K adapted to the actual image size.
    Calibration is given for 960x720, we scale if images are e.g. 640x480.
    """
    scale_x = width / CAMERA_WIDTH
    scale_y = height / CAMERA_HEIGHT

    fx = CAMERA_FX * scale_x
    fy = CAMERA_FY * scale_y
    cx = CAMERA_CX * scale_x
    cy = CAMERA_CY * scale_y

    print(f"[INFO] image size = ({width}x{height}), "
          f"scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    print("\nUsing camera intrinsics K =\n", K)
    return K


def detect_orb(img: np.ndarray, nfeatures: int = 3000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, ratio_thresh: float = 0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good


def rotation_matrix_to_euler_angles(R: np.ndarray):
    """
    Convert rotation matrix to yaw, pitch, roll (Z-Y-X convention).
    Returns (yaw, pitch, roll) in radians.
    """
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        yaw = math.atan2(R[1, 0], R[0, 0])      # around Z
        pitch = math.atan2(-R[2, 0], sy)        # around Y
        roll = math.atan2(R[2, 1], R[2, 2])     # around X
    else:
        yaw = math.atan2(-R[0, 1], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        roll = 0.0

    return yaw, pitch, roll


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def euler_rpy_deg_to_Rwc_gt(roll_deg, pitch_deg, yaw_deg):
    """
    Build rotation matrix Rwc for the GT (world_GT axes).
    Convention: Rwc = Rz(yaw) * Ry(pitch) * Rx(roll)
    Angles are in degrees from the simulator CSV.
    """
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

    Rwc_gt = Rz @ Ry @ Rx
    return Rwc_gt


# ==========================
# 3. POSE ESTIMATION
# ==========================

def estimate_pose_between_two_images(img1_gray, img2_gray, K):
    """
    Estimate relative pose (R, t) from img1 -> img2 using ORB + Essential + recoverPose.
    Returns:
        success (bool),
        R (3x3),
        t (3x1),
        C2 (3x1, camera 2 center in camera1 frame, up to scale),
        yaw_deg, pitch_deg, roll_deg (camera2 relative to camera1, in degrees)
    """
    kp1, des1 = detect_orb(img1_gray)
    kp2, des2 = detect_orb(img2_gray)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("[WARN] Not enough keypoints/descriptors.")
        return False, None, None, None, None, None, None

    good_matches = match_descriptors(des1, des2, ratio_thresh=0.75)
    print(f"[INFO] Good matches = {len(good_matches)}")
    if len(good_matches) < 8:
        print("[WARN] Not enough matches to estimate pose.")
        return False, None, None, None, None, None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    E, mask_E = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    if E is None:
        print("[WARN] Could not estimate Essential matrix.")
        return False, None, None, None, None, None, None

    print("\nEssential matrix E =\n", E)

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    print("\nRotation matrix R (cam2 relative to cam1) =\n", R)
    print("\nTranslation vector t (direction, up to scale) =\n", t)

    # Camera 2 center in camera1 frame
    C2 = -R.T @ t  # 3x1

    # Orientation as yaw, pitch, roll (camera2 relative to camera1)
    yaw, pitch, roll = rotation_matrix_to_euler_angles(R)
    yaw_deg = rad2deg(yaw)
    pitch_deg = rad2deg(pitch)
    roll_deg = rad2deg(roll)

    print("\nOrientation of camera 2 relative to camera 1 (degrees):")
    print(f"  yaw   = {yaw_deg:.3f} deg")
    print(f"  pitch = {pitch_deg:.3f} deg")
    print(f"  roll  = {roll_deg:.3f} deg")

    return True, R, t, C2, yaw_deg, pitch_deg, roll_deg


# ==========================
# 4. GROUND TRUTH (with alignment GT -> OpenCV)
# ==========================

def load_gt_aligned(frame1: int, frame2: int):
    """
    Load GT poses for two frames, and align them to OpenCV-like coordinates.
    GT axes (from your data analysis):
        +Z = forward
        +X = right
        +Y = up

    OpenCV camera/world axes:
        +Z = forward
        +X = right
        +Y = down

    So we need to flip Y:   v_cv = R_align @ v_gt,   with R_align = diag(1, -1, 1)
    For rotation matrices: Rwc_cv = R_align @ Rwc_gt @ R_align  (similarity transform).
    """
    if not POSES_CSV.exists():
        print(f"[WARN] GT file not found: {POSES_CSV}")
        return None, None

    df = pd.read_csv(POSES_CSV)

    row1 = df.loc[df["frame"] == frame1]
    row2 = df.loc[df["frame"] == frame2]

    if row1.empty or row2.empty:
        print(f"[WARN] GT rows not found for frames {frame1}, {frame2}")
        return None, None

    r1 = row1.iloc[0]
    r2 = row2.iloc[0]

    # GT positions (world_GT coordinates)
    p1_gt = np.array([float(r1["x"]), float(r1["y"]), float(r1["z"])])
    p2_gt = np.array([float(r2["x"]), float(r2["y"]), float(r2["z"])])

    # GT orientations (world_GT, in degrees)
    rpy1 = (float(r1["roll"]), float(r1["pitch"]), float(r1["yaw"]))
    rpy2 = (float(r2["roll"]), float(r2["pitch"]), float(r2["yaw"]))

    Rwc1_gt = euler_rpy_deg_to_Rwc_gt(*rpy1)
    Rwc2_gt = euler_rpy_deg_to_Rwc_gt(*rpy2)

    # Alignment GT -> OpenCV: flip Y
    R_align = np.diag([1.0, -1.0, 1.0])

    p1_cv = R_align @ p1_gt
    p2_cv = R_align @ p2_gt

    Rwc1_cv = R_align @ Rwc1_gt @ R_align
    Rwc2_cv = R_align @ Rwc2_gt @ R_align

    return (p1_cv, rpy1, Rwc1_cv), (p2_cv, rpy2, Rwc2_cv)


# ==========================
# 5. MAIN
# ==========================

def main():
    img1_path = IMAGES_DIR / f"{FRAME1:06d}.png"
    img2_path = IMAGES_DIR / f"{FRAME2:06d}.png"

    if not img1_path.exists() or not img2_path.exists():
        raise SystemExit(f"Images not found:\n  {img1_path}\n  {img2_path}")

    print(f"Using frames: {FRAME1} -> {FRAME2}")
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")

    # ---- load images ----
    img1 = load_gray_image(img1_path)
    img2 = load_gray_image(img2_path)

    h, w = img1.shape[:2]
    K = build_intrinsics(w, h)

    # ---- estimate pose from images ----
    ok, R, t, C2, yaw_est_rel, pitch_est_rel, roll_est_rel = estimate_pose_between_two_images(
        img1, img2, K
    )
    if not ok:
        return

    C2 = C2.flatten()
    print("\nEstimated camera positions in cam1 frame (up to scale):")
    print("  Cam1: (0.000000, 0.000000, 0.000000)")
    print(f"  Cam2: ({C2[0]:.6f}, {C2[1]:.6f}, {C2[2]:.6f})")

    # ---- load GT (aligned to OpenCV coordinates) ----
    print("\n=== Ground Truth (aligned GT -> OpenCV) ===")
    gt1, gt2 = load_gt_aligned(FRAME1, FRAME2)
    if gt1 is None or gt2 is None:
        return

    p1, rpy1, Rwc1 = gt1
    p2, rpy2, Rwc2 = gt2

    d_gt = p2 - p1

    print(f"GT frame {FRAME1}: pos=({p1[0]:.6f}, {p1[1]:.6f}, {p1[2]:.6f}), "
          f"rpy(deg)=({rpy1[0]:.3f}, {rpy1[1]:.3f}, {rpy1[2]:.3f})")
    print(f"GT frame {FRAME2}: pos=({p2[0]:.6f}, {p2[1]:.6f}, {p2[2]:.6f}), "
          f"rpy(deg)=({rpy2[0]:.3f}, {rpy2[1]:.3f}, {rpy2[2]:.3f})")

    print("\nGT relative translation (frame1 -> frame2) in WORLD (OpenCV-aligned):")
    print(f"  dpos = ({d_gt[0]:.6f}, {d_gt[1]:.6f}, {d_gt[2]:.6f})")

    # ===============================
    # 5a. Translation magnitude logic
    # ===============================
    est_dir = C2
    gt_dir = d_gt

    norm_est = np.linalg.norm(est_dir)
    norm_gt = np.linalg.norm(gt_dir)

    print("\n=== Translation norms ===")
    print(f"||EST translation|| (cam1 frame, up to scale) = {norm_est:.6f}")
    print(f"||GT  translation|| (world frame)             = {norm_gt:.6f}")

    ZERO_TOL = 1e-3

    if norm_gt < ZERO_TOL:
        print("\nGT says: NO MOTION between frames.")
        if norm_est < ZERO_TOL:
            print("EST also says: ~NO MOTION. ✅ (good)")
        else:
            print("EST says there IS motion. ❌ (expected ~0)")
        # No direction/scale comparison in this case
    elif norm_est < ZERO_TOL and norm_gt >= ZERO_TOL:
        print("\nEST says: ~NO MOTION, but GT shows real motion. ❌")
    else:
        # ===============================
        # 5b. Put EST in world frame + same scale
        # ===============================
        scale = norm_gt / norm_est
        C2_scaled = C2 * scale

        # --- קודם נבדוק כיוון בעולם, לפני שגוזרים את הפוז הסופי ---
        # תזוזה משוערת בעולם (לפני sign-fix):
        d_est_world = Rwc1 @ C2_scaled   # וקטור בין cam1->cam2 בעולם

        # כיוון GT בעולם:
        gt_dir_n = gt_dir / norm_gt
        est_dir_world_n = d_est_world / np.linalg.norm(d_est_world)

        dot_tmp = float(np.clip(np.dot(est_dir_world_n, gt_dir_n), -1.0, 1.0))
        angle_tmp = rad2deg(math.acos(dot_tmp))
        print(f"\n[DEBUG] angle between EST and GT directions (before sign-fix) = {angle_tmp:.3f} deg")

        # אם הכיוון רחוק מ-GT ביותר מ-90°, נהפוך סימן
        if angle_tmp > 90.0:
            print("[INFO] Flipping translation sign to match GT direction.")
            C2_scaled = -C2_scaled
            d_est_world = -d_est_world
            est_dir_world_n = -est_dir_world_n

        # עכשיו מחשבים את מיקום המצלמה השנייה בעולם לאחר sign-fix
        p2_est_world = p1 + d_est_world


        # world orientation of cam2 from EST:
        # cam2 in world = Rwc1 * R (cam1->cam2)
        Rwc2_est = Rwc1 @ R

        # rotation error between Rwc2_est and GT Rwc2
        R_err = Rwc2_est.T @ Rwc2
        trace_val = np.trace(R_err)
        val = max(min((trace_val - 1.0) / 2.0, 1.0), -1.0)
        angle_err_rad = math.acos(val)
        angle_err_deg = rad2deg(angle_err_rad)

        # EST world orientation as yaw,pitch,roll
        yaw_w, pitch_w, roll_w = rotation_matrix_to_euler_angles(Rwc2_est)
        yaw_w_deg = rad2deg(yaw_w)
        pitch_w_deg = rad2deg(pitch_w)
        roll_w_deg = rad2deg(roll_w)

        print("\n=== EST pose in WORLD (OpenCV-aligned), scaled to GT ===")
        print(f"EST frame {FRAME2}: pos=({p2_est_world[0]:.6f}, {p2_est_world[1]:.6f}, {p2_est_world[2]:.6f}), "
              f"rpy(deg)=({roll_w_deg:.3f}, {pitch_w_deg:.3f}, {yaw_w_deg:.3f})")

        # Pose error (world frame)
        pos_err = p2_est_world - p2
        pos_err_norm = np.linalg.norm(pos_err)

        print("\nPose error in WORLD (OpenCV-aligned):")
        print(f"  position error = ({pos_err[0]:.6f}, {pos_err[1]:.6f}, {pos_err[2]:.6f}), "
              f"|err|={pos_err_norm:.6f}")
        print(f"  rotation error angle = {angle_err_deg:.3f} deg")

        # Also show directions in world frame
        est_dir_world = p2_est_world - p1
        if np.linalg.norm(est_dir_world) > 1e-9 and norm_gt > 1e-9:
            est_dir_world_n = est_dir_world / np.linalg.norm(est_dir_world)
            gt_dir_n = gt_dir / norm_gt
            dot_w = float(np.clip(np.dot(est_dir_world_n, gt_dir_n), -1.0, 1.0))
            ang_w = rad2deg(math.acos(dot_w))
            print("\nDirection comparison in WORLD frame:")
            print(f"  unit(EST dpos_world) = {est_dir_world_n}")
            print(f"  unit(GT  dpos_world) = {gt_dir_n}")
            print(f"  angle between directions = {ang_w:.3f} deg")


if __name__ == "__main__":
    main()
