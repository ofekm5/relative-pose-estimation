import cv2
import numpy as np
import math
import argparse
from pathlib import Path


# ==========================
# 1. CONFIG
# ==========================

BASE_DIR = Path("../silmulator_data")
IMAGES_DIR = BASE_DIR / "images"
UNDIST_DIR = BASE_DIR / "images_undist"

UNDIST_DIR.mkdir(parents=True, exist_ok=True)

# Default frames (can be overridden by CLI)
DEFAULT_FRAME1 = 0
DEFAULT_FRAME2 = 90

# Original calibration (960x720)
CAMERA_FX = 924.82939686
CAMERA_FY = 920.4766382
CAMERA_CX = 468.24930789
CAMERA_CY = 353.65863024
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

# Distortion coefficients from simulator
K1 = 2.34795296e-02
K2 = -4.38293524e-01
P1 = -1.48496922e-03
P2 = -8.11809445e-06


# ==========================
# 2. UTILS
# ==========================

def build_intrinsics_and_dist(width: int, height: int):
    """
    Build camera matrix K adapted to the actual image size,
    and distortion coefficients vector for OpenCV.
    """
    scale_x = width / CAMERA_WIDTH
    scale_y = height / CAMERA_HEIGHT

    fx = CAMERA_FX * scale_x
    fy = CAMERA_FY * scale_y
    cx = CAMERA_CX * scale_x
    cy = CAMERA_CY * scale_y

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    distCoeffs = np.array([K1, K2, P1, P2, 0.0], dtype=np.float64)

    print(f"[INFO] image size = ({width}x{height}), "
          f"scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
    print("\nUsing camera intrinsics K =\n", K)
    print("\nUsing distortion coeffs =\n", distCoeffs)

    return K, distCoeffs


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


def detect_orb(img_gray: np.ndarray, nfeatures: int = 3000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, ratio_thresh: float = 0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good


# ==========================
# 3. MAIN PIPELINE
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Estimate relative pose between two images with undistortion."
    )
    parser.add_argument("--frame1", type=int, default=DEFAULT_FRAME1,
                        help="First frame index")
    parser.add_argument("--frame2", type=int, default=DEFAULT_FRAME2,
                        help="Second frame index")
    args = parser.parse_args()

    frame1 = args.frame1
    frame2 = args.frame2

    img1_path = IMAGES_DIR / f"{frame1:06d}.png"
    img2_path = IMAGES_DIR / f"{frame2:06d}.png"

    if not img1_path.exists() or not img2_path.exists():
        raise SystemExit(f"Images not found:\n  {img1_path}\n  {img2_path}")

    print(f"Using frames: {frame1} -> {frame2}")
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")

    # ---- load color images (for saving) ----
    img1_color = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
    img2_color = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)

    if img1_color is None or img2_color is None:
        raise SystemExit("Failed to load images in color.")

    h, w = img1_color.shape[:2]
    K, distCoeffs = build_intrinsics_and_dist(w, h)

    # ---- undistort images ----
    img1_undist = cv2.undistort(img1_color, K, distCoeffs)
    img2_undist = cv2.undistort(img2_color, K, distCoeffs)

    # Save undistorted images
    out1 = UNDIST_DIR / f"{frame1:06d}_undist.png"
    out2 = UNDIST_DIR / f"{frame2:06d}_undist.png"
    cv2.imwrite(str(out1), img1_undist)
    cv2.imwrite(str(out2), img2_undist)
    print(f"[INFO] Saved undistorted images:\n  {out1}\n  {out2}")

    # Convert to grayscale for ORB
    img1_gray = cv2.cvtColor(img1_undist, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_undist, cv2.COLOR_BGR2GRAY)

    # ---- feature detection & matching ----
    kp1, des1 = detect_orb(img1_gray)
    kp2, des2 = detect_orb(img2_gray)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("[WARN] Not enough keypoints/descriptors.")
        return

    good_matches = match_descriptors(des1, des2, ratio_thresh=0.75)
    print(f"[INFO] Good matches = {len(good_matches)}")
    if len(good_matches) < 8:
        print("[WARN] Not enough matches to estimate pose.")
        return

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # ---- Essential matrix ----
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
        return

    print("\nEssential matrix E =\n", E)

    # ---- Recover pose ----
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    print("\nRotation matrix R (cam2 relative to cam1) =\n", R)
    print("\nTranslation vector t (direction, up to scale) =\n", t)

    # Camera 2 center in camera1 frame
    C2 = -R.T @ t
    C2 = C2.flatten()

    # Orientation
    yaw, pitch, roll = rotation_matrix_to_euler_angles(R)
    yaw_deg = rad2deg(yaw)
    pitch_deg = rad2deg(pitch)
    roll_deg = rad2deg(roll)

    print("\nOrientation of camera 2 relative to camera 1 (degrees):")
    print(f"  yaw   = {yaw_deg:.3f} deg")
    print(f"  pitch = {pitch_deg:.3f} deg")
    print(f"  roll  = {roll_deg:.3f} deg")

    print("\nEstimated camera positions in cam1 frame (up to scale):")
    print("  Cam1: (0.000000, 0.000000, 0.000000)")
    print(f"  Cam2: ({C2[0]:.6f}, {C2[1]:.6f}, {C2[2]:.6f})")


if __name__ == "__main__":
    main()
