import os
import cv2
import numpy as np
import pandas as pd

def load_dataset(base_dir):
    """
    base_dir:
        simulatorOutputDir on C++ side
        ├── images/
        │   ├── 000000.png
        │   ├── 000001.png
        │   └── ...
        └── camera_poses.csv
    """
    images_dir = os.path.join(base_dir, "images")
    poses_path = os.path.join(base_dir, "camera_poses.csv")

    # read GT file
    poses_df = pd.read_csv(poses_path)

    # sort frames by index
    poses_df = poses_df.sort_values("frame")

    data = []

    for _, row in poses_df.iterrows():
        frame_idx = int(row["frame"])
        img_name = f"{frame_idx:06d}.png"
        img_path = os.path.join(images_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] missing image for frame {frame_idx}: {img_path}")
            continue

        pose = {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "roll": float(row["roll"]),
            "pitch": float(row["pitch"]),
            "yaw": float(row["yaw"]),
        }

        data.append({
            "frame": frame_idx,
            "image": img,
            "pose": pose
        })

    return data


if __name__ == "__main__":
    base_dir = "../silmulator_data"
    dataset = load_dataset(base_dir)
    print(f"Loaded {len(dataset)} frames")
    # example: show first image + its GT
    if dataset:
        print("First frame:", dataset[0]["frame"], dataset[0]["pose"])
        cv2.imshow("first image", dataset[0]["image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
