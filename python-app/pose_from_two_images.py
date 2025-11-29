import cv2
import numpy as np
from pathlib import Path
import math


# ---------- Utils ----------

def load_gray_image(path: str) -> np.ndarray:
    """Load image from disk and convert to grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def detect_orb(img: np.ndarray, nfeatures: int = 3000):
    """Detect ORB keypoints and descriptors."""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def match_descriptors(des1, des2, ratio_thresh: float = 0.75):
    """
    Match two sets of ORB descriptors using BFMatcher + Lowe ratio test.
    Returns list of good matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches


def rotation_matrix_to_euler_angles(R: np.ndarray):
    """
    Convert rotation matrix to Euler angles (yaw, pitch, roll) in radians.
    Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    yaw   = rotation around Z (azimuth)
    pitch = rotation around Y
    roll  = rotation around X
    """
    assert R.shape == (3, 3)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        yaw = math.atan2(R[1, 0], R[0, 0])          # around Z
        pitch = math.atan2(-R[2, 0], sy)            # around Y
        roll = math.atan2(R[2, 1], R[2, 2])         # around X
    else:
        # Gimbal lock case
        yaw = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        roll = 0.0

    return yaw, pitch, roll


def rad2deg(x):
    return x * 180.0 / math.pi


# ---------- Main pipeline ----------

def main():
    # ---- 1. Set your image paths here ----
    img1_path = "../images/img1.png"   # first view
    img2_path = "../images/img2.png"   # second view

    if not Path(img1_path).exists() or not Path(img2_path).exists():
        raise SystemExit("Update img1_path/img2_path to valid image files")

    # ---- 2. Load images ----
    img1 = load_gray_image(img1_path)
    img2 = load_gray_image(img2_path)

    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")

    # ---- 3. Detect ORB features ----
    kp1, des1 = detect_orb(img1, nfeatures=3000)
    kp2, des2 = detect_orb(img2, nfeatures=3000)

    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        raise SystemExit("Not enough keypoints/descriptors found in images")

    # ---- 4. Match descriptors ----
    good_matches = match_descriptors(des1, des2, ratio_thresh=0.75)
    print(f"Good matches: {len(good_matches)}")

    if len(good_matches) < 8:
        raise SystemExit("Not enough good matches to estimate pose")

    # ---- 5. Build matched point arrays ----
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # ---- 6. Camera intrinsics (approximate if unknown) ----
    h, w = img1.shape[:2]

    fx = fy = 900.0   # you can change if you know real intrinsics
    cx = w / 2.0
    cy = h / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    print("\nIntrinsic matrix K =\n", K)

    # ---- 7. Estimate Essential matrix ----
    E, mask_E = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        raise SystemExit("Could not estimate Essential matrix")

    print("\nEssential matrix E =\n", E)

    # ---- 8. Recover pose (R, t) ----
    # R, t describe the transform from camera1 to camera2
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    print("\nRotation matrix R (cam2 relative to cam1) =\n", R)
    print("\nTranslation vector t (direction, up to scale) =\n", t)

    # ---- 9. Compute camera 2 position in world (camera 1 frame) ----
    # World frame = camera 1 frame:
    # Camera 1: C1 = (0, 0, 0)
    # Camera 2 center: C2 = -R^T * t
    C2 = -R.T @ t  # 3x1 vector
    x, y, z = C2.flatten().tolist()

    print("\nCamera positions in world frame (up to scale):")
    print(f" Camera 1: (0.0, 0.0, 0.0)")
    print(f" Camera 2: (X={x:.6f}, Y={y:.6f}, Z={z:.6f})")

    # ---- 10. Convert R to Euler angles (azimuth, pitch, roll) ----
    yaw, pitch, roll = rotation_matrix_to_euler_angles(R)

    azimuth_deg = rad2deg(yaw)
    pitch_deg = rad2deg(pitch)
    roll_deg = rad2deg(roll)

    print("\nOrientation of camera 2 relative to camera 1:")
    print(" (radians)")
    print(f"  azimuth (yaw) = {yaw:.6f}")
    print(f"  pitch         = {pitch:.6f}")
    print(f"  roll          = {roll:.6f}")

    print("\n (degrees)")
    print(f"  azimuth (yaw) = {azimuth_deg:.3f} deg")
    print(f"  pitch         = {pitch_deg:.3f} deg")
    print(f"  roll          = {roll_deg:.3f} deg")

    # =============================================================
    # 11. Approximate covariance using bootstrap over inlier matches
    # =============================================================

    inlier_mask = mask_pose.ravel().astype(bool)
    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]

    n_inliers = pts1_in.shape[0]
    print(f"\nInliers after recoverPose: {n_inliers}")

    if n_inliers < 15:
        print("Not enough inliers for reasonable covariance estimation.")
        samples = None
    else:
        N_BOOT = 100  # you can increase this for smoother covariance
        rng = np.random.default_rng(seed=42)

        samples_list = []

        for b in range(N_BOOT):
            # Sample inliers with replacement
            idx = rng.integers(0, n_inliers, size=n_inliers)
            p1b = pts1_in[idx]
            p2b = pts2_in[idx]

            Eb, _ = cv2.findEssentialMat(
                p1b,
                p2b,
                K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            if Eb is None:
                continue

            try:
                _, Rb, tb, _ = cv2.recoverPose(Eb, p1b, p2b, K)
            except cv2.error:
                continue

            C2b = -Rb.T @ tb
            xb, yb, zb = C2b.flatten().tolist()
            yawb, pitchb, rollb = rotation_matrix_to_euler_angles(Rb)

            samples_list.append([xb, yb, zb, yawb, pitchb, rollb])

        if len(samples_list) < 10:
            print("Bootstrap produced too few valid samples, covariance will be unreliable.")
            samples = None
        else:
            samples = np.array(samples_list)  # shape: (N, 6)

    if samples is not None:
        mean_params = samples.mean(axis=0)
        cov_params = np.cov(samples, rowvar=False)

        print("\n=== Bootstrap pose statistics (up to scale) ===")
        print("Parameters order: [X, Y, Z, yaw, pitch, roll]")

        print("\nMean parameters:")
        print(f" X    = {mean_params[0]:.6f}")
        print(f" Y    = {mean_params[1]:.6f}")
        print(f" Z    = {mean_params[2]:.6f}")
        print(f" yaw  = {mean_params[3]:.6f} rad ({rad2deg(mean_params[3]):.3f} deg)")
        print(f" pitch= {mean_params[4]:.6f} rad ({rad2deg(mean_params[4]):.3f} deg)")
        print(f" roll = {mean_params[5]:.6f} rad ({rad2deg(mean_params[5]):.3f} deg)")

        print("\nCovariance matrix (6x6):")
        print(cov_params)

        std_params = np.sqrt(np.diag(cov_params))
        print("\nStandard deviations:")
        print(f" sigma_X    = {std_params[0]:.6f}")
        print(f" sigma_Y    = {std_params[1]:.6f}")
        print(f" sigma_Z    = {std_params[2]:.6f}")
        print(f" sigma_yaw  = {std_params[3]:.6f} rad ({rad2deg(std_params[3]):.3f} deg)")
        print(f" sigma_pitch= {std_params[4]:.6f} rad ({rad2deg(std_params[4]):.3f} deg)")
        print(f" sigma_roll = {std_params[5]:.6f} rad ({rad2deg(std_params[5]):.3f} deg)")
    else:
        print("\nCovariance could not be reliably estimated (too few valid bootstrap samples).")

    # ---- 12. (Optional) show matches ----
    show_matches = True
    if show_matches:
        matched_vis = cv2.drawMatches(
            img1, kp1,
            img2, kp2,
            good_matches[:50],   # show up to 50 matches
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow("Good matches", matched_vis)
        print("\nPress any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
