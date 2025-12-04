import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

from gt_utils import load_gt_poses, evaluate_pair


# ---------------------------------------------------------
# הגדרות
# ---------------------------------------------------------

base_dir = Path("../silmulator_data/simple_movement")
poses_path = base_dir / "camera_poses.txt"

df_gt = load_gt_poses(poses_path)

# טווח פריימים לבדיקה
START = 0
END   = 500
STEP  = 15

# סף חריגות לשגיאת מיקום
OUTLIER_POS_TH = 0.25   # 25 ס״מ


# ---------------------------------------------------------
# הרצה על כל זוגות הפריימים
# ---------------------------------------------------------

pairs = []
pos_errors = []
roll_errors = []
pitch_errors = []
yaw_errors = []

for f1 in range(START, END, STEP):
    f2 = f1 + STEP

    try:
        pos_err, d_roll, d_pitch, d_yaw = evaluate_pair(base_dir, df_gt, f1, f2)
    except Exception as e:
        print(f"FAILED on pair {f1}->{f2}: {e}")
        continue

    pairs.append((f1, f2))
    pos_errors.append(pos_err)
    roll_errors.append(d_roll)
    pitch_errors.append(d_pitch)
    yaw_errors.append(d_yaw)

    print(f"{f1:04d}->{f2:04d} | pos_err={pos_err:.3f} m | "
          f"roll={d_roll:.2f}°, pitch={d_pitch:.2f}°, yaw={d_yaw:.2f}°")


# ---------------------------------------------------------
# חישוב אחוז שגיאה (יחס ל־GT)
# ---------------------------------------------------------

percentage_errors = []

for (f1, f2), pos_err in zip(pairs, pos_errors):
    row1 = df_gt[df_gt["frame"] == f1].iloc[0]
    row2 = df_gt[df_gt["frame"] == f2].iloc[0]

    gt_vec = np.array([
        row2.x - row1.x,
        row2.y - row1.y,
        row2.z - row1.z
    ], dtype=float)

    gt_dist = np.linalg.norm(gt_vec)

    if gt_dist < 1e-9:
        percentage_errors.append(np.nan)
    else:
        pct = 100.0 * pos_err / gt_dist
        percentage_errors.append(pct)


# ---------------------------------------------------------
# הכנה לגרפים
# ---------------------------------------------------------

pos_errors        = np.array(pos_errors)
percentage_errors = np.array(percentage_errors)
roll_errors       = np.array(roll_errors)
pitch_errors      = np.array(pitch_errors)
yaw_errors        = np.array(yaw_errors)

indices   = np.arange(len(pos_errors))
is_outlier = pos_errors > OUTLIER_POS_TH

# צבע לכל נקודה לפי אם היא חריגה או לא
colors = np.where(is_outlier, "red", "blue")


# =========================================================
#   שלושה גרפים: מיקום, אחוז, זוויות (עם hover)
# =========================================================

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# ---- גרף 1: שגיאת מיקום ----
ax1.set_title("Position Error Per Frame Pair (scaled, cam1 frame)")
ax1.set_ylabel("Position Error [m]")
ax1.grid(True)

sc_pos = ax1.scatter(indices, pos_errors, c=colors)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Normal',
           markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label=f'Outlier (>{OUTLIER_POS_TH} m)',
           markerfacecolor='red', markersize=8),
]
ax1.legend(handles=legend_elements, loc="upper right")

# ---- גרף 2: אחוז שגיאה ----
ax2.set_title("Relative Position Error (percentage)")
ax2.set_ylabel("Error [%]")
ax2.grid(True)

sc_pct = ax2.scatter(indices, percentage_errors, c=colors)

# ---- גרף 3: שגיאות זווית ----
ax3.set_title("Angle Errors Per Frame Pair (scatter)")
ax3.set_xlabel("Pair index")
ax3.set_ylabel("Error [deg]")
ax3.grid(True)

sc_roll  = ax3.scatter(indices, roll_errors,  label="Roll error")
sc_pitch = ax3.scatter(indices, pitch_errors, label="Pitch error")
sc_yaw   = ax3.scatter(indices, yaw_errors,   label="Yaw error")

ax3.legend(loc="upper right")

plt.tight_layout()


# ---------------------------------------------------------
# HOVER: תוויות לכל אחד משלושת הגרפים
# ---------------------------------------------------------

# annotation לכל ציר
annot1 = ax1.annotate(
    "",
    xy=(0, 0),
    xytext=(15, 15),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->"),
)
annot1.set_visible(False)

annot2 = ax2.annotate(
    "",
    xy=(0, 0),
    xytext=(15, 15),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->"),
)
annot2.set_visible(False)

annot3 = ax3.annotate(
    "",
    xy=(0, 0),
    xytext=(15, 15),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->"),
)
annot3.set_visible(False)


def update_annot_pos(idx):
    x = indices[idx]
    y = pos_errors[idx]
    annot1.xy = (x, y)

    f1, f2 = pairs[idx]
    pos_err_val = pos_errors[idx]
    pct_err_val = percentage_errors[idx]

    text = (
        f"{f1:04d} -> {f2:04d}\n"
        f"pos_err = {pos_err_val:.3f} m\n"
        f"rel_err = {pct_err_val:.1f} %"
    )
    annot1.set_text(text)
    annot1.get_bbox_patch().set_alpha(0.9)


def update_annot_pct(idx):
    x = indices[idx]
    y = percentage_errors[idx]
    annot2.xy = (x, y)

    f1, f2 = pairs[idx]
    pos_err_val = pos_errors[idx]
    pct_err_val = percentage_errors[idx]

    text = (
        f"{f1:04d} -> {f2:04d}\n"
        f"pos_err = {pos_err_val:.3f} m\n"
        f"rel_err = {pct_err_val:.1f} %"
    )
    annot2.set_text(text)
    annot2.get_bbox_patch().set_alpha(0.9)


def update_annot_ang(idx):
    x = indices[idx]
    # נבחר y כלשהו, למשל yaw, רק בשביל מיקום החץ
    y = yaw_errors[idx]
    annot3.xy = (x, y)

    f1, f2 = pairs[idx]
    text = (
        f"{f1:04d} -> {f2:04d}\n"
        f"roll_err  = {roll_errors[idx]:.2f}°\n"
        f"pitch_err = {pitch_errors[idx]:.2f}°\n"
        f"yaw_err   = {yaw_errors[idx]:.2f}°"
    )
    annot3.set_text(text)
    annot3.get_bbox_patch().set_alpha(0.9)


def hide_all_annots():
    annot1.set_visible(False)
    annot2.set_visible(False)
    annot3.set_visible(False)


def hover(event):
    # אם לא על אחד מהצירים – מסתירים הכל
    if event.inaxes not in (ax1, ax2, ax3):
        if annot1.get_visible() or annot2.get_visible() or annot3.get_visible():
            hide_all_annots()
            fig.canvas.draw_idle()
        return

    # מחשבים אינדקס לפי xdata (קרוב לנקודה)
    if event.xdata is None:
        return

    idx = int(round(event.xdata))
    if idx < 0 or idx >= len(indices):
        return

    # ציר עליון – pos error
    if event.inaxes == ax1:
        hide_all_annots()
        update_annot_pos(idx)
        annot1.set_visible(True)
        fig.canvas.draw_idle()

    # ציר אמצעי – percentage error
    elif event.inaxes == ax2:
        hide_all_annots()
        update_annot_pct(idx)
        annot2.set_visible(True)
        fig.canvas.draw_idle()

    # ציר תחתון – angle errors
    elif event.inaxes == ax3:
        hide_all_annots()
        update_annot_ang(idx)
        annot3.set_visible(True)
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
