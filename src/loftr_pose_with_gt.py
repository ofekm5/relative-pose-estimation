import cv2
import numpy as np
import math
import pandas as pd
import torch
import kornia as K
import kornia.feature as KF
from pathlib import Path

# ==========================
# 1. CONFIGURATION
# ==========================

BASE_DIR = Path("../silmulator_data")   # adjust if needed
IMAGES_DIR = BASE_DIR / "images"
POSES_CSV = BASE_DIR / "camera_poses.csv"

# Default frames to compare (row indices in camera_poses.csv
# AND image indices 000XXX.png)
FRAME1 = 1520
FRAME2 = 1530

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

def build_intrinsics(width: int, height: int) -> np.ndarray:
    """
    Build camera intrinsics matrix K for the given image size, based on
    original calibration at 960x720.
    """
    scale_x = width / CAMERA_WIDTH
    scale_y = height / CAMERA_HEIGHT

    fx = CAMERA_FX * scale_x
    fy = CAMERA_FY * scale_y
    cx = CAMERA_CX * scale_x
    cy = CAMERA_CY * scale_y

    Kmat = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    print(f"[INFO] image size = ({width}x{height}), scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
    print("\nUsing camera intrinsics K =\n", Kmat)
    return Kmat


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


def euler_z_y_x_to_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Build rotation matrix from yaw(Z), pitch(Y), roll(X) in radians.
    R = Rz(yaw) * Ry(pitch) * Rx(roll).
    """
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    Rz = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ])
    Ry = np.array([
        [ cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  cr, -sr],
        [0.0,  sr,  cr],
    ])
    return Rz @ Ry @ Rx


def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def load_image_as_torch_gray(path: Path, device: torch.device):
    """
    Load image as grayscale and convert to torch tensor (1,1,H,W) in [0,1].
    Also return the original numpy grayscale.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    t = K.image_to_tensor(img, keepdim=False).float() / 255.0  # (1,1,H,W)
    return t.to(device), img


def r2ypr(R: np.ndarray):
    """
    Rotation matrix -> (yaw, pitch, roll) in degrees, Z-Y-X convention.
    """
    yaw, pitch, roll = rotation_matrix_to_euler_angles(R)
    return rad2deg(yaw), rad2deg(pitch), rad2deg(roll)


def compare_rotations(R_est: np.ndarray, R_gt: np.ndarray):
    """
    Compare two rotation matrices:
    - Compute rotation angle error.
    - Compute yaw/pitch/roll errors (deg) in Z-Y-X convention.
    """
    R_err = R_gt @ R_est.T
    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = math.acos(trace)
    angle_deg = rad2deg(angle_rad)

    yaw_est, pitch_est, roll_est = r2ypr(R_est)
    yaw_gt, pitch_gt, roll_gt = r2ypr(R_gt)

    yaw_err = yaw_est - yaw_gt
    pitch_err = pitch_est - pitch_gt
    roll_err = roll_est - roll_gt

    return angle_deg, yaw_err, pitch_err, roll_err


# ==========================
# 3. GROUND TRUTH HANDLING
# ==========================

def read_gt_pose(row_idx: int):
    """
    Read GT position and yaw/pitch/roll (in degrees) for given ROW index
    from camera_poses.csv.

    Assumes CSV format (no header):
      frame, x, y, z, yaw_deg, pitch_deg, roll_deg

    And we align to OpenCV-like world by flipping y -> -y.
    """
    df = pd.read_csv(POSES_CSV, header=None)
    df = df.iloc[:, :7]
    df.columns = ["frame", "x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg"]

    if row_idx < 0 or row_idx >= len(df):
        raise ValueError(f"Row index {row_idx} out of range for {POSES_CSV}")

    row = df.iloc[row_idx]
    frame_val = int(row["frame"])
    print(f"[INFO] read_gt_pose: requested row_idx={row_idx}, frame value in CSV={frame_val}")

    x = float(row["x"])
    y = float(row["y"])
    z = float(row["z"])
    yaw_deg = float(row["yaw_deg"])
    pitch_deg = float(row["pitch_deg"])
    roll_deg = float(row["roll_deg"])

    # Align world so that y is flipped
    pos_world = np.array([x, -y, z], dtype=np.float64)

    yaw = deg2rad(yaw_deg)
    pitch = deg2rad(pitch_deg)
    roll = deg2rad(roll_deg)

    # R_world_cam: world -> camera
    R_world_cam = euler_z_y_x_to_rotation(yaw, pitch, roll)

    return pos_world, np.array([yaw_deg, pitch_deg, roll_deg]), R_world_cam


# ==========================
# 4. PURE ROTATION CHECK
# ==========================

def is_almost_pure_rotation(Kmat: np.ndarray,
                            pts1: np.ndarray,
                            pts2: np.ndarray,
                            R: np.ndarray,
                            threshold_px: float = 10.0,
                            min_inliers: int = 50) -> bool:
    """
    Heuristic: check if the motion between two views can be explained
    almost entirely by a pure rotation (no translation), using only images.

    Kmat: 3x3 intrinsics
    pts1, pts2: Nx2 matching points (float32)
    R: 3x3 rotation from recoverPose (cam2 relative to cam1)
    threshold_px: average residual (in pixels) below which we say it's "almost pure rotation"
    """
    if pts1.shape[0] < min_inliers:
        return False

    Kinv = np.linalg.inv(Kmat)

    ones = np.ones((pts1.shape[0], 1), dtype=np.float64)
    pts1_h = np.hstack([pts1.astype(np.float64), ones])  # (N,3)

    # Rays in camera1 frame
    rays1 = (Kinv @ pts1_h.T).T  # (N,3)

    # Apply rotation only
    rays2_rot = (R @ rays1.T).T  # (N,3)

    x = rays2_rot[:, 0] / rays2_rot[:, 2]
    y = rays2_rot[:, 1] / rays2_rot[:, 2]

    proj_rot_h = np.stack([x, y, np.ones_like(x)], axis=1)  # (N,3)
    proj_rot = (Kmat @ proj_rot_h.T).T  # (N,3)

    u2_rot = proj_rot[:, 0]
    v2_rot = proj_rot[:, 1]

    du = pts2[:, 0] - u2_rot
    dv = pts2[:, 1] - v2_rot
    residuals = np.sqrt(du * du + dv * dv)

    mean_res = float(np.mean(residuals))
    print(f"[DEBUG] pure-rotation check: mean residual = {mean_res:.3f} px over {pts1.shape[0]} points")

    return mean_res < threshold_px


# ==========================
# 5. POSE ESTIMATION WITH LoFTR
# ==========================

def estimate_pose_loftr(frame1: int, frame2: int):
    """
    Use LoFTR to match features between two images, estimate Essential matrix
    and then relative pose (R,t) and camera center C2 (in cam1 frame).
    Also performs a pure-rotation check and zeros t if needed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    img1_path = IMAGES_DIR / f"{frame1:06d}.png"
    img2_path = IMAGES_DIR / f"{frame2:06d}.png"

    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")

    timg0, img0 = load_image_as_torch_gray(img1_path, device)
    timg1, img1 = load_image_as_torch_gray(img2_path, device)

    h, w = img0.shape[:2]
    Kmat = build_intrinsics(w, h)

    # Initialize LoFTR
    print("[INFO] Loading LoFTR (pretrained='indoor')...")
    matcher = KF.LoFTR(pretrained="indoor").to(device)
    matcher.eval()

    with torch.no_grad():
        batch = {"image0": timg0, "image1": timg1}
        output = matcher(batch)

    kpts0 = output["keypoints0"].cpu().numpy()  # Nx2
    kpts1 = output["keypoints1"].cpu().numpy()  # Nx2

    num_matches = kpts0.shape[0]
    print(f"[INFO] LoFTR produced {num_matches} matches.")
    if num_matches < 8:
        print("[WARN] Not enough matches to estimate pose.")
        return False, None, None, None, None, None, None, None, None, None

    pts1 = kpts0.astype(np.float32)
    pts2 = kpts1.astype(np.float32)

    # Essential matrix
    E, mask_E = cv2.findEssentialMat(
        pts1, pts2, Kmat,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    if E is None:
        print("[WARN] Could not estimate Essential matrix.")
        return False, None, None, None, None, None, None, None, None, None

    print("\nEssential matrix E =\n", E)

    # Relative pose
    _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts1, pts2, Kmat)
    print("\nRotation matrix R (cam2 relative to cam1) =\n", R_rel)
    print("\nTranslation vector t (direction, up to scale) =\n", t_rel)

    # Pure rotation check
    if is_almost_pure_rotation(Kmat, pts1, pts2, R_rel, threshold_px=10.0):
        print("\n[INFO] Detected almost pure rotation – translation is unreliable.")
        print("[INFO] Setting translation t to zero vector.")
        t_rel = np.zeros_like(t_rel)

    # Camera 2 center in camera1 frame
    C2_cam1 = -R_rel.T @ t_rel  # 3x1

    # Yaw/pitch/roll of R_rel
    yaw, pitch, roll = rotation_matrix_to_euler_angles(R_rel)
    yaw_deg = rad2deg(yaw)
    pitch_deg = rad2deg(pitch)
    roll_deg = rad2deg(roll)

    print("\nOrientation of camera 2 relative to camera 1 (degrees):")
    print(f"  yaw   = {yaw_deg:.3f} deg")
    print(f"  pitch = {pitch_deg:.3f} deg")
    print(f"  roll  = {roll_deg:.3f} deg")

    return True, R_rel, t_rel, C2_cam1, yaw_deg, pitch_deg, roll_deg, pts1, pts2, Kmat


# ==========================
# 6. MAIN: COMPARE ESTIMATE VS GT
# ==========================

def main():
    frame1 = FRAME1
    frame2 = FRAME2

    # --- Estimate pose with LoFTR ---
    (
        ok,
        R_rel_est,
        t_rel_est,
        C2_cam1,
        yaw_deg,
        pitch_deg,
        roll_deg,
        pts1,
        pts2,
        Kmat
    ) = estimate_pose_loftr(frame1, frame2)

    if not ok:
        return

    # --- Load GT for both frames ---
    print("\n=== Ground Truth (aligned GT -> OpenCV) ===")
    p1_world, rpy1_deg, R_world_cam1 = read_gt_pose(frame1)
    p2_world, rpy2_deg, R_world_cam2 = read_gt_pose(frame2)

    d_gt = p2_world - p1_world

    print(f"GT frame {frame1}: pos=({p1_world[0]:.6f}, {p1_world[1]:.6f}, {p1_world[2]:.6f}), "
          f"rpy(deg)=({rpy1_deg[0]:.3f}, {rpy1_deg[1]:.3f}, {rpy1_deg[2]:.3f})")
    print(f"GT frame {frame2}: pos=({p2_world[0]:.6f}, {p2_world[1]:.6f}, {p2_world[2]:.6f}), "
          f"rpy(deg)=({rpy2_deg[0]:.3f}, {rpy2_deg[1]:.3f}, {rpy2_deg[2]:.3f})")

    print("\nGT relative translation (frame1 -> frame2) in WORLD (OpenCV-aligned):")
    print(f"  dpos = ({d_gt[0]:.6f}, {d_gt[1]:.6f}, {d_gt[2]:.6f})")

    # --- Translation norms ---
    est_vec_cam1 = C2_cam1.flatten()
    gt_vec_world = d_gt

    norm_est = np.linalg.norm(est_vec_cam1)
    norm_gt = np.linalg.norm(gt_vec_world)

    print("\n=== Translation norms ===")
    print(f"||EST translation|| (cam1 frame, up to scale) = {norm_est:.6f}")
    print(f"||GT  translation|| (world frame)             = {norm_gt:.6f}")

    ZERO_TOL = 1e-3

    # --- Handle zero-motion GT case (pure rotation) ---
    if norm_gt < ZERO_TOL and norm_est < ZERO_TOL:
        print("\nGT says: NO MOTION between frames.")
        print("EST also says: ~NO MOTION. ✅ (good)")
        # Still compare rotations:
        R_rel_gt = R_world_cam2 @ R_world_cam1.T
        angle_deg, dy, dp, dr = compare_rotations(R_rel_est, R_rel_gt)
        print("\n=== ROTATION ERROR vs GT (relative R) ===")
        print(f"Rotation angle error: {angle_deg:.3f} deg")
        print(f"Yaw error:   {dy:.3f} deg")
        print(f"Pitch error: {dp:.3f} deg")
        print(f"Roll error:  {dr:.3f} deg")
        return

    if norm_gt < ZERO_TOL and norm_est >= ZERO_TOL:
        print("\nGT says: NO MOTION, but EST has motion. ❌ (bad for translation).")
        # Compare rotations anyway
        R_rel_gt = R_world_cam2 @ R_world_cam1.T
        angle_deg, dy, dp, dr = compare_rotations(R_rel_est, R_rel_gt)
        print("\n=== ROTATION ERROR vs GT (relative R) ===")
        print(f"Rotation angle error: {angle_deg:.3f} deg")
        print(f"Yaw error:   {dy:.3f} deg")
        print(f"Pitch error: {dp:.3f} deg")
        print(f"Roll error:  {dr:.3f} deg")
        return

    if norm_gt >= ZERO_TOL and norm_est < ZERO_TOL:
        print("\nEST says: NO MOTION, but GT has motion. ❌")
        # Still compare rotations:
        R_rel_gt = R_world_cam2 @ R_world_cam1.T
        angle_deg, dy, dp, dr = compare_rotations(R_rel_est, R_rel_gt)
        print("\n=== ROTATION ERROR vs GT (relative R) ===")
        print(f"Rotation angle error: {angle_deg:.3f} deg")
        print(f"Yaw error:   {dy:.3f} deg")
        print(f"Pitch error: {dp:.3f} deg")
        print(f"Roll error:  {dr:.3f} deg")
        return

    # --- Non-zero motion case: scale & compare translation ---
    scale = norm_gt / norm_est
    est_scaled_cam1 = est_vec_cam1 * scale

    # Direction check in CAM1 frame
    gt_dir_cam1 = R_world_cam1.T @ gt_vec_world
    est_dir_cam1 = est_vec_cam1 / norm_est
    gt_dir_cam1_n = gt_dir_cam1 / np.linalg.norm(gt_dir_cam1)

    dot_cam = float(np.clip(np.dot(est_dir_cam1, gt_dir_cam1_n), -1.0, 1.0))
    ang_cam = rad2deg(math.acos(dot_cam))
    print(f"\n[DEBUG] angle between EST and GT directions in CAM1 frame (before sign-fix) = {ang_cam:.3f} deg")

    # Flip sign if needed
    if ang_cam > 90.0:
        print("[INFO] Flipping translation sign to match GT direction.")
        est_scaled_cam1 *= -1.0

    # EST pose in WORLD
    p1 = p1_world
    est_d_world = R_world_cam1 @ est_scaled_cam1
    p2_est_world = p1 + est_d_world

    pos_err = p2_est_world - p2_world
    pos_err_norm = np.linalg.norm(pos_err)

    print("\n=== EST pose in WORLD (OpenCV-aligned), scaled to GT ===")
    print(f"EST frame {frame2}: pos=({p2_est_world[0]:.6f}, {p2_est_world[1]:.6f}, {p2_est_world[2]:.6f})")

    print("\nPose error in WORLD (OpenCV-aligned):")
    print(f"  position error = ({pos_err[0]:.6f}, {pos_err[1]:.6f}, {pos_err[2]:.6f}), |err|={pos_err_norm:.6f}")

    # --- Rotation error vs GT (relative rotation) ---
    R_rel_gt = R_world_cam2 @ R_world_cam1.T
    angle_deg, dy, dp, dr = compare_rotations(R_rel_est, R_rel_gt)

    print(f"  rotation angle error (relative R) = {angle_deg:.3f} deg")

    # --- Direction comparison in WORLD frame ---
    est_dir_world_n = est_d_world / np.linalg.norm(est_d_world)
    gt_dir_world_n = gt_vec_world / norm_gt
    dot_w = float(np.clip(np.dot(est_dir_world_n, gt_dir_world_n), -1.0, 1.0))
    ang_w = rad2deg(math.acos(dot_w))
    print("\nDirection comparison in WORLD frame:")
    print(f"  unit(EST dpos_world) = {est_dir_world_n}")
    print(f"  unit(GT  dpos_world) = {gt_dir_world_n}")
    print(f"  angle between directions = {ang_w:.3f} deg")

    print("\n=== ROTATION ERROR vs GT (relative R) ===")
    print(f"Yaw error:   {dy:.3f} deg")
    print(f"Pitch error: {dp:.3f} deg")
    print(f"Roll error:  {dr:.3f} deg")


if __name__ == "__main__":
    main()
