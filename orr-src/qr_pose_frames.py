#!/usr/bin/env python3
import os
import glob
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil


# ==========================
# 1) EDIT ONLY THESE
# ==========================
FRAMES_DIR = "//home/orr/university_projects/qr_position/forward_with_stuff"
OUT_DIR    = "/home/orr/university_projects/qr_position/forward_with_stuff_analysis"
K_NPZ_PATH = "/home/orr/university_projects/relative-pose-estimation/src/calibration_filtered.npz"

# IMPORTANT: the resolution that the calibration K belongs to (before scaling)
CALIB_W, CALIB_H = 2000, 1126   # <-- change if your calibration images were different

TAG_ID = 0
TAG_SIZE_M = 0.20               # 200mm
ERR_MAX_PX = 0.1
FILTERED_FRAMES_DIR = os.path.join(OUT_DIR, "filtered_frames")

# Recommended for planar square: IPPE_SQUARE
PNP_FLAG = cv2.SOLVEPNP_ITERATIVE # or cv2.SOLVEPNP_ITERATIVE

FORCE_ZERO_DIST = False
WRAP_ANGLES_180 = False
# ==========================

def scatter_series(out_png: str, xvals: np.ndarray, series: List[Tuple[str, np.ndarray]], title: str):
    plt.figure(figsize=(10, 5))
    for name, y in series:
        plt.scatter(xvals, y, s=10, label=name)  # s=10 = נקודות קטנות
    plt.title(title)
    plt.xlabel("Frame index")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def scale_K_to_frame(K, frame_w, frame_h, calib_w, calib_h):
    sx = frame_w / float(calib_w)
    sy = frame_h / float(calib_h)
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx  # fx
    K2[1, 1] *= sy  # fy
    K2[0, 2] *= sx  # cx
    K2[1, 2] *= sy  # cy
    return K2


def wrap180(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def rotmat_to_az_pitch_roll_deg_camera(R: np.ndarray) -> Tuple[float, float, float]:
    # ZYX (yaw-pitch-roll)
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0

    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def make_tag_objp(tag_size_m: float) -> np.ndarray:
    s = tag_size_m
    return np.array([
        [-s/2, -s/2, 0.0],  # TL
        [ s/2, -s/2, 0.0],  # TR
        [ s/2,  s/2, 0.0],  # BR
        [-s/2,  s/2, 0.0],  # BL
    ], dtype=np.float64)


@dataclass
class Row:
    frame_idx: int
    filepath: str
    filename: str
    x: float
    y: float
    z: float
    az: float
    pitch: float
    roll: float
    bearing: float
    elevation: float
    reproj_err: float


def list_images(frames_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(frames_dir, e)))
    files.sort()
    return files


def detect_tag_pose(img_bgr, detector, K, dist, objp, target_id: int, pnp_flag: int):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    ids = ids.flatten().astype(int)
    keep = np.where(ids == target_id)[0]
    if len(keep) == 0:
        return None

    i = int(keep[0])
    corners = corners_list[i].reshape(4, 2).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=pnp_flag)
    if not ok:
        return None

    # keep tag in front of camera
    if float(tvec[2, 0]) < 0.0:
        rvec = -rvec
        tvec = -tvec

    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    reproj_err = float(np.mean(np.linalg.norm(proj - corners, axis=1)))

    return rvec, tvec, reproj_err


def plot_series(out_png: str, xvals: np.ndarray, series: List[Tuple[str, np.ndarray]], title: str):
    plt.figure(figsize=(10, 5))
    for name, y in series:
        plt.plot(xvals, y, label=name)
    plt.title(title)
    plt.xlabel("Frame index")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    files = list_images(FRAMES_DIR)
    if not files:
        raise RuntimeError(f"No images found in: {FRAMES_DIR}")

    sample = cv2.imread(files[0])
    if sample is None:
        raise RuntimeError(f"Failed to read: {files[0]}")
    h, w = sample.shape[:2]

    if not os.path.isfile(K_NPZ_PATH):
        raise RuntimeError(f"Missing calibration npz: {K_NPZ_PATH}")

    data = np.load(K_NPZ_PATH)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64) if "dist" in data.files else np.zeros((5, 1), dtype=np.float64)
    dist = np.zeros((1,5))
    # Scale K if frame size differs from calibration size
    if (w, h) != (CALIB_W, CALIB_H):
        K = scale_K_to_frame(K, frame_w=w, frame_h=h, calib_w=CALIB_W, calib_h=CALIB_H)

    if FORCE_ZERO_DIST:
        dist = np.zeros((5, 1), dtype=np.float64)

    objp = make_tag_objp(TAG_SIZE_M)

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    os.makedirs(OUT_DIR, exist_ok=True)

    rows: List[Row] = []
    missed = 0

    for idx, fname in enumerate(files):
        img = cv2.imread(fname)
        if img is None:
            missed += 1
            continue

        pose = detect_tag_pose(img, detector, K, dist, objp, TAG_ID, PNP_FLAG)
        if pose is None:
            missed += 1
            continue

        rvec, tvec, reproj = pose
        R_tc, _ = cv2.Rodrigues(rvec)

        x, y, z = tvec.flatten().astype(float)
        az, pitch, roll = rotmat_to_az_pitch_roll_deg_camera(R_tc)

        bearing = math.degrees(math.atan2(x, z))
        elevation = math.degrees(math.atan2(-y, z))

        if WRAP_ANGLES_180:
            az = wrap180(az)
            pitch = wrap180(pitch)
            roll = wrap180(roll)

        rows.append(Row(
            idx,
            fname,
            os.path.basename(fname),
            x, y, z,
            az, pitch, roll,
            bearing, elevation,
            reproj
        ))
        rows_good = [r for r in rows if r.reproj_err < ERR_MAX_PX]
        # Create/clean folder for filtered frames
        os.makedirs(FILTERED_FRAMES_DIR, exist_ok=True)

        # Optional: clean previous content
        for old in glob.glob(os.path.join(FILTERED_FRAMES_DIR, "*")):
            try:
                os.remove(old)
            except:
                pass

        # Copy filtered frames
        for r in rows_good:
            dst = os.path.join(FILTERED_FRAMES_DIR, r.filename)
            shutil.copy2(r.filepath, dst)

        print(f"[DONE] Copied {len(rows_good)} filtered frames to: {FILTERED_FRAMES_DIR}")
        print(f"[FILTER] reproj_err < {ERR_MAX_PX} px")
        print(f"[FILTER] kept: {len(rows_good)} / {len(rows)}")
    # Save CSV
    out_csv_all = os.path.join(OUT_DIR, "tag0_pose_all.csv")
    with open(out_csv_all, "w", encoding="utf-8") as f:
        f.write("frame_idx,filename,x,y,z,az,pitch,roll,bearing,elevation,reproj_err_px\n")
        for r in rows:
            f.write(f"{r.frame_idx},{r.filename},"
                    f"{r.x:.6f},{r.y:.6f},{r.z:.6f},"
                    f"{r.az:.3f},{r.pitch:.3f},{r.roll:.3f},"
                    f"{r.bearing:.3f},{r.elevation:.3f},"
                    f"{r.reproj_err:.3f}\n")

    # Save CSV (FILTERED)
    out_csv = os.path.join(OUT_DIR, "tag0_pose_filtered.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("frame_idx,filename,x,y,z,az,pitch,roll,bearing,elevation,reproj_err_px\n")
        for r in rows_good:
            f.write(f"{r.frame_idx},{r.filename},"
                    f"{r.x:.6f},{r.y:.6f},{r.z:.6f},"
                    f"{r.az:.3f},{r.pitch:.3f},{r.roll:.3f},"
                    f"{r.bearing:.3f},{r.elevation:.3f},"
                    f"{r.reproj_err:.3f}\n")

    print(f"[DONE] CSV ALL:      {out_csv_all}")
    print(f"[DONE] CSV FILTERED: {out_csv}")

    if len(rows_good) < 2:
        print("[WARN] Not enough filtered detections to plot.")
        return

    frame_idx = np.array([r.frame_idx for r in rows_good], dtype=float)
    xs = np.array([r.x for r in rows_good], dtype=float)
    ys = np.array([r.y for r in rows_good], dtype=float)
    zs = np.array([r.z for r in rows_good], dtype=float)
    azs = np.array([r.az for r in rows_good], dtype=float)
    azs_unwrapped = np.degrees(np.unwrap(np.radians(azs)))
    ps = np.array([r.pitch for r in rows_good], dtype=float)
    rs = np.array([r.roll for r in rows_good], dtype=float)
    bear = np.array([r.bearing for r in rows_good], dtype=float)
    elev = np.array([r.elevation for r in rows_good], dtype=float)
    err = np.array([r.reproj_err for r in rows_good], dtype=float)

    scatter_series(os.path.join(OUT_DIR, "position_xyz_vs_frame.png"),
                   frame_idx, [("x (m)", xs), ("y (m)", ys), ("z (m)", zs)],
                   f"TAG{TAG_ID} Position (Camera Frame)")

    scatter_series(os.path.join(OUT_DIR, "angles_az_pitch_roll_vs_frame.png"),
                   frame_idx,
                   [("az_unwrapped (deg)", azs_unwrapped), ("pitch (deg)", ps), ("roll (deg)", rs)],
                   f"TAG{TAG_ID} Euler Angles ZYX (Camera Frame)")

    scatter_series(os.path.join(OUT_DIR, "bearing_elevation_vs_frame.png"),
                   frame_idx, [("bearing atan2(x,z) (deg)", bear), ("elevation atan2(-y,z) (deg)", elev)],
                   f"TAG{TAG_ID} Bearing/Elevation from tvec")

    scatter_series(os.path.join(OUT_DIR, "reprojection_error_vs_frame.png"),
                   frame_idx, [("reproj err (px)", err)],
                   f"TAG{TAG_ID} Reprojection Error")

    print(f"[DONE] Plots saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
