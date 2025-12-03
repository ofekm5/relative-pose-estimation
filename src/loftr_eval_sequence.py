import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# נייבא מהסקריפט הקודם
from loftr_pose_with_gt import (
    estimate_pose_loftr,
    read_gt_pose,
    compare_rotations,
    POSES_CSV,
    rad2deg,
)

ZERO_TOL = 1e-3   # סף להחלטה "אין תנועה"

# ספים "נורמליים" לשגיאות
POS_THRESHOLD = 0.10   # meters (10cm)
ROT_THRESHOLD = 2.0    # degrees


def compute_errors_over_sequence(step: int = 20):
    """
    רץ על כל הרצף בקפיצות של 'step' (למשל 20):
      (0 -> 20), (20 -> 40), ...
    מחזיר:
      pair_f1:     רשימת הפריים הראשון בזוג
      pair_f2:     רשימת הפריים השני בזוג
      pair_center: אינדקס "מרכזי" לציור (למשל f2)
      pos_errors:  שגיאת מיקום (נורמה, במטרים)
      rot_errors:  שגיאת זווית (angle error, במעלות)
    """

    df = pd.read_csv(POSES_CSV, header=None)
    num_rows = len(df)

    pair_f1 = []
    pair_f2 = []
    pair_centers = []
    pos_errors = []
    rot_errors = []

    for f1 in range(0, num_rows - step, step):
        f2 = f1 + step

        print(f"\n=== Evaluating pair: {f1} -> {f2} ===")

        # 1) הערכת פוזה מהתמונות (LoFTR)
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
        ) = estimate_pose_loftr(f1, f2)

        if not ok or R_rel_est is None or C2_cam1 is None:
            print("[WARN] Pose estimation failed for this pair. Skipping.")
            continue

        # 2) GT עבור שני הפריימים
        p1_world, rpy1_deg, R_world_cam1 = read_gt_pose(f1)
        p2_world, rpy2_deg, R_world_cam2 = read_gt_pose(f2)

        d_gt = p2_world - p1_world
        norm_gt = np.linalg.norm(d_gt)

        # 3) שגיאת רוטציה (יחסית)
        R_rel_gt = R_world_cam2 @ R_world_cam1.T
        angle_deg, dy, dp, dr = compare_rotations(R_rel_est, R_rel_gt)

        # 4) שגיאת תרגום (נורמה, לאחר סקלון ויישור כיוון)
        est_vec_cam1 = C2_cam1.flatten()
        norm_est = np.linalg.norm(est_vec_cam1)

        if norm_gt < ZERO_TOL and norm_est < ZERO_TOL:
            # אין תנועה אמיתית וגם האלגוריתם אומר שאין תנועה
            pos_err_norm = 0.0
        elif norm_gt < ZERO_TOL and norm_est >= ZERO_TOL:
            # GT אומר אין תנועה, האלגוריתם אומר שיש
            pos_err_norm = norm_est
        elif norm_gt >= ZERO_TOL and norm_est < ZERO_TOL:
            # GT אומר שיש תנועה, האלגוריתם אומר אין
            pos_err_norm = norm_gt
        else:
            # שני הצדדים אומרים שיש תנועה -> נעשה סקלון והשוואה כמו בסקריפט הקודם
            scale = norm_gt / norm_est
            est_scaled_cam1 = est_vec_cam1 * scale

            # יישור כיוון לפי world
            gt_vec_world = d_gt
            gt_dir_cam1 = R_world_cam1.T @ gt_vec_world
            est_dir_cam1 = est_vec_cam1 / norm_est
            gt_dir_cam1_n = gt_dir_cam1 / np.linalg.norm(gt_dir_cam1)

            dot_cam = float(np.clip(np.dot(est_dir_cam1, gt_dir_cam1_n), -1.0, 1.0))
            ang_cam = rad2deg(np.arccos(dot_cam))

            if ang_cam > 90.0:
                est_scaled_cam1 *= -1.0

            # האומדן בעולם
            est_d_world = R_world_cam1 @ est_scaled_cam1
            p2_est_world = p1_world + est_d_world

            pos_err = p2_est_world - p2_world
            pos_err_norm = np.linalg.norm(pos_err)

        # נשמור את התוצאות
        pair_f1.append(f1)
        pair_f2.append(f2)
        pair_centers.append(f2)          # אפשר גם (f1+f2)/2
        pos_errors.append(pos_err_norm)
        rot_errors.append(angle_deg)

        print(f"[RESULT] pair {f1}->{f2}: pos_err = {pos_err_norm:.4f} m, rot_err = {angle_deg:.4f} deg")

    return (
        np.array(pair_f1),
        np.array(pair_f2),
        np.array(pair_centers),
        np.array(pos_errors),
        np.array(rot_errors),
    )


def plot_errors(pair_f1, pair_f2, pair_centers, pos_errors, rot_errors):
    """
    מצייר שני גרפים:
      1) שגיאת מיקום
      2) שגיאת זווית
    עם סימון נקודות חריגות באדום (לפי סף קבוע),
    ותוויות אינטראקטיביות עם העכבר שמציגות את צמד הפריימים.
    """

    # ננסה לייבא mplcursors (ל-tooltip עם העכבר)
    try:
        import mplcursors
        have_mplcursors = True
    except ImportError:
        print("[WARN] mplcursors not installed. Run: pip install mplcursors")
        have_mplcursors = False

    # --- Position error outliers ---
    pos_is_outlier = pos_errors > POS_THRESHOLD

    # --- Rotation error outliers ---
    rot_is_outlier = rot_errors > ROT_THRESHOLD

    # ========= Plot Position Error =========
    fig1 = plt.figure(figsize=(10, 4))
    ax1 = fig1.add_subplot(111)

    scat_pos_normal = ax1.scatter(
        pair_centers[~pos_is_outlier],
        pos_errors[~pos_is_outlier],
        label="Position error (normal)",
    )
    scat_pos_out = ax1.scatter(
        pair_centers[pos_is_outlier],
        pos_errors[pos_is_outlier],
        color="red",
        label="Position error (outlier)",
    )

    ax1.axhline(
        POS_THRESHOLD,
        linestyle="--",
        label=f"Threshold = {POS_THRESHOLD*100:.0f} cm"
    )

    ax1.set_xlabel("Frame index (second frame in pair)")
    ax1.set_ylabel("Position error [m]")
    ax1.set_title("Position Error vs Frame (LoFTR vs GT)")
    ax1.legend()
    ax1.grid(True)

    # ========= Plot Rotation Error =========
    fig2 = plt.figure(figsize=(10, 4))
    ax2 = fig2.add_subplot(111)

    scat_rot_normal = ax2.scatter(
        pair_centers[~rot_is_outlier],
        rot_errors[~rot_is_outlier],
        label="Rotation error (normal)",
    )
    scat_rot_out = ax2.scatter(
        pair_centers[rot_is_outlier],
        rot_errors[rot_is_outlier],
        color="red",
        label="Rotation error (outlier)",
    )

    ax2.axhline(
        ROT_THRESHOLD,
        linestyle="--",
        label=f"Threshold = {ROT_THRESHOLD:.1f}°"
    )

    ax2.set_xlabel("Frame index (second frame in pair)")
    ax2.set_ylabel("Rotation error [deg]")
    ax2.set_title("Rotation Error vs Frame (LoFTR vs GT)")
    ax2.legend()
    ax2.grid(True)

    # ========= Tooltips עם העכבר =========
    if have_mplcursors:
        # Position plot
        def _make_pos_label(index):
            f1 = pair_f1[index]
            f2 = pair_f2[index]
            pe = pos_errors[index]
            re = rot_errors[index]
            return f"{f1} -> {f2}\npos_err = {pe:.3f} m\nrot_err = {re:.3f}°"

        # נחבר גם לנורמליים וגם לחריגים
        cursor_pos = mplcursors.cursor(
            [scat_pos_normal, scat_pos_out],
            hover=True
        )

        @cursor_pos.connect("add")
        def on_add_pos(sel):
            idx = sel.index
            sel.annotation.set(text=_make_pos_label(idx))

        # Rotation plot
        def _make_rot_label(index):
            f1 = pair_f1[index]
            f2 = pair_f2[index]
            pe = pos_errors[index]
            re = rot_errors[index]
            return f"{f1} -> {f2}\npos_err = {pe:.3f} m\nrot_err = {re:.3f}°"

        cursor_rot = mplcursors.cursor(
            [scat_rot_normal, scat_rot_out],
            hover=True
        )

        @cursor_rot.connect("add")
        def on_add_rot(sel):
            idx = sel.index
            sel.annotation.set(text=_make_rot_label(idx))

    plt.tight_layout()
    plt.show()


def main():
    # נריץ על כל הרצף בקפיצות של 20 פריימים
    pair_f1, pair_f2, pair_centers, pos_errors, rot_errors = compute_errors_over_sequence(step=20)

    # נצייר את הגרפים
    plot_errors(pair_f1, pair_f2, pair_centers, pos_errors, rot_errors)


if __name__ == "__main__":
    main()
