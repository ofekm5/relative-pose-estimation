import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pose_from_two_images import load_gt_poses, evaluate_pair


def main():
    base_dir = Path("../silmulator_data/simple_movement")
    poses_path = base_dir / "camera_poses.txt"

    # טוענים GT פעם אחת
    df_gt = load_gt_poses(poses_path)

    # קפיצה בין פריימים (אפשר לשנות אם תרצה)
    step = 15

    # כל הפריימים שיש ב-GT
    frames = sorted(df_gt["frame"].unique())
    frame_set = set(frames)

    starts = []       # פריים התחלה לכל קפיצה
    pos_errors = []   # שגיאת מיקום (נורמה)
    roll_errors = []  # |Δroll|
    pitch_errors = [] # |Δpitch|
    yaw_errors = []   # |Δyaw|

    for f in frames:
        f2 = f + step
        if f2 not in frame_set:
            continue  # אין פריים כזה, מדלגים

        try:
            pos_err, d_roll, d_pitch, d_yaw = evaluate_pair(base_dir, df_gt, f, f2)
        except Exception as e:
            print(f"Skipping pair {f}->{f2}: {e}")
            continue

        starts.append(f)
        pos_errors.append(pos_err)
        roll_errors.append(abs(d_roll))
        pitch_errors.append(abs(d_pitch))
        yaw_errors.append(abs(d_yaw))

    if not starts:
        print("No valid frame pairs found.")
        return

    x = np.array(starts)

    # -------- גרף 1: שגיאת מיקום (נקודות) --------
    plt.figure()
    plt.scatter(x, pos_errors)
    plt.xlabel("Start frame")
    plt.ylabel("Position error |Δ| [m]")
    plt.title(f"Position error for jumps of {step} frames")

    # -------- גרף 2: שגיאות זווית (נקודות) --------
    plt.figure()
    plt.scatter(x, roll_errors,  label="|Δroll|")
    plt.scatter(x, pitch_errors, label="|Δpitch|")
    plt.scatter(x, yaw_errors,   label="|Δyaw|")
    plt.xlabel("Start frame")
    plt.ylabel("Angle error [deg]")
    plt.title(f"Angle errors for jumps of {step} frames")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
