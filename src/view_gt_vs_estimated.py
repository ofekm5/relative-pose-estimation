import os
import cv2
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# we reuse your helpers from pose_from_two_images.py
from pose_from_two_images import (
    detect_orb,
    match_descriptors,
    rotation_matrix_to_euler_angles,
    rad2deg,
)

# ========= CONFIG =========
BASE_DIR = "../silmulator_data"  # TODO: change to your simulatorOutputDir
IMAGES_DIR = os.path.join(BASE_DIR, "images")
POSES_PATH = os.path.join(BASE_DIR, "camera_poses.csv")
STEP = 1        # use every frame pair (k-1, k)
DELAY = 1     # seconds between frames in the viewer

# ---- Camera parameters from simulator ----
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
# ==========================


def load_poses():
    df = pd.read_csv(POSES_PATH)
    df = df.sort_values("frame").reset_index(drop=True)
    return df


def build_intrinsics(width, height):
    """
    Build camera intrinsics K using the REAL simulator calibration.
    """
    # sanity check – just to be sure we use the expected resolution
    if width != CAMERA_WIDTH or height != CAMERA_HEIGHT:
        print(f"[WARN] image size ({width}x{height}) != camera calibration ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")

    K = np.array([
        [CAMERA_FX,        0.0,       CAMERA_CX],
        [0.0,        CAMERA_FY,       CAMERA_CY],
        [0.0,              0.0,            1.0 ],
    ], dtype=np.float64)

    return K



def estimate_relative_pose(img1_gray, img2_gray, K):
    """
    Run the SAME pipeline as in pose_from_two_images:
    ORB -> match -> Essential -> recoverPose.
    Returns:
        success (bool),
        C2_est (3,),  # camera 2 center in camera 1 frame (up to scale)
        roll_deg, pitch_deg, yaw_deg
    """
    # 1. detect ORB
    kp1, des1 = detect_orb(img1_gray, nfeatures=3000)
    kp2, des2 = detect_orb(img2_gray, nfeatures=3000)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("[WARN] not enough keypoints/descriptors")
        return False, None, None, None, None

    # 2. match
    good_matches = match_descriptors(des1, des2, ratio_thresh=0.75)
    print(f"Good matches: {len(good_matches)}")
    if len(good_matches) < 8:
        print("[WARN] not enough good matches to estimate pose")
        return False, None, None, None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # 3. Essential
    E, mask_E = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    if E is None:
        print("[WARN] cannot estimate Essential matrix")
        return False, None, None, None, None

    # 4. recover pose
    try:
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    except cv2.error as e:
        print("[WARN] recoverPose failed:", e)
        return False, None, None, None, None

    # Camera 2 center in camera 1 frame: C2 = -R^T t
    C2 = -R.T @ t
    C2 = C2.flatten()

    # orientation (same convention as in your file)
    yaw, pitch, roll = rotation_matrix_to_euler_angles(R)
    yaw_deg = rad2deg(yaw)
    pitch_deg = rad2deg(pitch)
    roll_deg = rad2deg(roll)

    return True, C2, roll_deg, pitch_deg, yaw_deg


def main():
    poses_df = load_poses()
    n_rows = len(poses_df)
    print(f"Loaded {n_rows} GT rows from {POSES_PATH}")

    if n_rows < 2:
        print("Not enough frames to compare.")
        return

    # we will display the SECOND image (frame k) for each pair (k-1, k)
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    img_artist = None
    text_artist = None

    # loop over consecutive pairs
    for idx in range(1, n_rows, STEP):
        row1 = poses_df.iloc[idx - 1]
        row2 = poses_df.iloc[idx]

        f1 = int(row1["frame"])
        f2 = int(row2["frame"])

        img1_path = os.path.join(IMAGES_DIR, f"{f1:06d}.png")
        img2_path = os.path.join(IMAGES_DIR, f"{f2:06d}.png")

        # load images
        img1_gray = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2_bgr = cv2.imread(img2_path, cv2.IMREAD_COLOR)
        img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

        img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

        if img1_gray is None or img2_gray is None:
            print(f"[WARN] missing images for frames {f1} or {f2}")
            continue

        h, w = img1_gray.shape[:2]
        K = build_intrinsics(w, h)

        # estimate relative pose (algorithm result)
        ok, C2_est, roll_est, pitch_est, yaw_est = estimate_relative_pose(
            img1_gray, img2_gray, K
        )

        # GT positions (absolute, simulator world)
        x1, y1, z1 = float(row1["x"]), float(row1["y"]), float(row1["z"])
        x2, y2, z2 = float(row2["x"]), float(row2["y"]), float(row2["z"])

        roll2_gt = float(row2["roll"])
        pitch2_gt = float(row2["pitch"])
        yaw2_gt = float(row2["yaw"])

        # build text
        if ok:
            # algorithm translation is up to scale; we can also normalize if you like
            Xe, Ye, Ze = C2_est.tolist()
            txt = (
                f"frames: {f1} -> {f2}\n"
                f"GT p1: {x1:.3f}  {y1:.3f}  {z1:.3f}\n"
                f"GT p2: {x2:.3f}  {y2:.3f}  {z2:.3f}\n"
                f"EST p2 (rel, up to scale): {Xe:.3f}  {Ye:.3f}  {Ze:.3f}\n"
                f"GT rpy2 (deg): roll={roll2_gt:.2f}  pitch={pitch2_gt:.2f}  yaw={yaw2_gt:.2f}\n"
                f"EST rpy (deg): roll={roll_est:.2f}  pitch={pitch_est:.2f}  yaw={yaw_est:.2f}"
            )
        else:
            txt = (
                f"frames: {f1} -> {f2}\n"
                f"GT p1: {x1:.3f}  {y1:.3f}  {z1:.3f}\n"
                f"GT p2: {x2:.3f}  {y2:.3f}  {z2:.3f}\n"
                f"[POSE ESTIMATION FAILED]"
            )

        # update plot
        ax.clear()
        ax.imshow(img2_rgb)
        ax.axis("off")
        ax.set_title("Frame k image + GT vs Estimated pose", fontsize=11)

        ax.text(
            0.02,
            0.95,
            txt,
            color="yellow",
            fontsize=9,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
        )

        plt.draw()
        plt.pause(DELAY)

    print("Done – close the window to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
