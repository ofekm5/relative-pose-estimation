# gt_utils.py
import numpy as np
import pandas as pd
import re


def clean_header(name: str):
    """
    Clean a header:
    - remove spaces
    - remove parentheses
    - lower-case everything
    Example:
        ' pitch (deg) ' -> 'pitch'
        '   x  '       -> 'x'
    """
    name = name.strip()
    name = re.sub(r"\s+", "", name)      # remove all whitespace
    name = re.sub(r"\(.*?\)", "", name)  # remove parentheses
    return name.lower()


def load_gt_poses(path):
    """
    Load GT poses (camera_poses.txt).
    Expected real columns:
       frame, x, y, z, roll, pitch, yaw
    """

    df = pd.read_csv(path, sep=r"\s+", engine="python")

    # clean headers (just in case)
    df.columns = [clean_header(c) for c in df.columns]

    # keep only the correct fields
    expected = ["frame", "x", "y", "z", "roll", "pitch", "yaw"]
    df = df[expected]

    return df


def evaluate_pair(base_dir, df_gt, frame1, frame2):
    """
    Compute the GT relative position and Euler angle deltas.
    """

    row1 = df_gt[df_gt["frame"] == frame1].iloc[0]
    row2 = df_gt[df_gt["frame"] == frame2].iloc[0]

    # Translation vector GT:
    t1 = np.array([row1.x, row1.y, row1.z])
    t2 = np.array([row2.x, row2.y, row2.z])
    dt = t2 - t1

    pos_err = np.linalg.norm(dt)

    # Euler deltas:
    dr = row2.roll  - row1.roll
    dp = row2.pitch - row1.pitch
    dy = row2.yaw   - row1.yaw

    return pos_err, dr, dp, dy
