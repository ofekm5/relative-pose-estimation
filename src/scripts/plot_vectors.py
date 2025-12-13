import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Use relative path from project root
data_path = project_root / "data" / "camera_poses.txt"
df = pd.read_csv(data_path, delim_whitespace=True)
df_sub = df[df["frame"] % 15 == 0]

# לוקחים רק את הנקודות במרחב
x = df_sub["x"].values
y = df_sub["y"].values
z = df_sub["z"].values
frames = df_sub["frame"].values

fig = go.Figure()

# מסלול המצלמה (קו + נקודות)
fig.add_trace(
    go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines+markers",
        line=dict(color="blue", width=4),
        marker=dict(size=2, color="blue"),
        text=[f"frame: {f}" for f in frames],  # ← hover text
        hovertemplate="Frame %{text}<extra></extra>",
        name="camera path",
    )
)

fig.update_layout(
    title="Displacement vectors between points (p1→p2, p2→p3)",
    width=900,
    height=700,
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        xaxis=dict(range=[-5, 5]),
        yaxis=dict(range=[-5, 5]),
        zaxis=dict(range=[-5, 5]),
        aspectmode="data",
    ),
    showlegend=True,
)

fig.show()
