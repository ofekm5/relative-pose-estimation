import numpy as np
import pandas as pd
import plotly.graph_objects as go



def build_path_from_rpy(roll, pitch, yaw, step_len=0.05):
    positions = [np.array([0.0, 0.0, 0.0])]
    for r, p, yw in zip(roll, pitch, yaw):
        d = rpy_to_direction(r, p, yw)
        new_pos = positions[-1] + step_len * d
        positions.append(new_pos)
    return np.array(positions)


# --------------------------------------------------------
# 1. פונקציה: RPY -> וקטור כיוון (אורך 1)
# --------------------------------------------------------
def rpy_to_direction(roll_deg, pitch_deg, yaw_deg):
    roll  = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw   = np.deg2rad(yaw_deg)

    # סיבוב סביב Y (yaw)
    R_yaw = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [ 0,           1, 0          ],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # סיבוב סביב X (pitch)
    R_pitch = np.array([
        [1, 0,            0           ],
        [0, np.cos(pitch),-np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    # סיבוב סביב Z (roll)
    R_roll = np.array([
        [ np.cos(roll), -np.sin(roll), 0],
        [ np.sin(roll),  np.cos(roll), 0],
        [ 0,              0,           1]
    ])

    # סדר הסיבובים (Y-up)
    R = R_yaw @ R_pitch @ R_roll

    # מצביע קדימה
    base = np.array([0, 0, 1], dtype=float)

    d = R @ base
    return d / np.linalg.norm(d)


# --------------------------------------------------------
# 2. טוענים את הדאטה → רק פריימים שהם כפולות של 15
# --------------------------------------------------------
df = pd.read_csv("/home/orr/university_projects/relative-pose-estimation/silmulator_data/simple_movement/camera_poses.txt", delim_whitespace=True)
df_sub = df[df["frame"] % 15 == 0].reset_index(drop=True)

roll  = df_sub["roll"].values
pitch = df_sub["pitch"].values
yaw   = df_sub["yaw"].values
frames = df_sub["frame"].values

# --------------------------------------------------------
# 3. מסלול על סמך זוויות בלבד
# --------------------------------------------------------
step_len = 0.05  # מרחק בין נקודות (תשחק עם זה אם תרצה)

positions = [np.array([0.0, 0.0, 0.0])]  # מיקום התחלתי

for r, p, yw in zip(roll, pitch, yaw):
    d = rpy_to_direction(r, p, yw)
    new_pos = positions[-1] + step_len * d
    positions.append(new_pos)

positions = np.array(positions)

x = positions[:, 0]
y = positions[:, 1]
z = positions[:, 2]

# --------------------------------------------------------
# 4. ציור הגרף ב-Plotly
# --------------------------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines+markers",
        line=dict(color="blue", width=5),
        marker=dict(size=4, color="red"),
        text=[f"frame: {f}" for f in [-1] + list(frames)],
        hovertemplate="%{text}<extra></extra>",
        name="Path from RPY only",
    )
)

fig.update_layout(
    title="3D Path from Roll/Pitch/Yaw Only (sampled every 15 frames)",
    width=1300,
    height=900,
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",

        # שומר על קובייה – כולם באותו סקייל
        aspectmode="cube",

        # זווית מצלמה התחלתית נוחה לתלת־מימד
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.2),  # מאיפה "מצלמים"
            up=dict(x=0, y=0, z=1)          # איפה למעלה
        ),
    ),
    showlegend=True
)

fig.show()
