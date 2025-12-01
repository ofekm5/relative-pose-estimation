import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIG =========
BASE_DIR = "../silmulator_data"  # <- CHANGE THIS
IMAGES_DIR = os.path.join(BASE_DIR, "images")
POSES_PATH = os.path.join(BASE_DIR, "camera_poses.csv")
STEP = 1        # how many frames to skip each time (1 = every frame)
DELAY = 0.1     # seconds between frames in the viewer
# ==========================


def load_dataset(base_dir):
    images_dir = os.path.join(base_dir, "images")
    poses_path = os.path.join(base_dir, "camera_poses.csv")

    poses_df = pd.read_csv(poses_path)
    poses_df = poses_df.sort_values("frame")

    records = []
    for _, row in poses_df.iterrows():
        frame_idx = int(row["frame"])
        img_name = f"{frame_idx:06d}.png"
        img_path = os.path.join(images_dir, img_name)

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] missing image for frame {frame_idx}: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        records.append(
            {
                "frame": frame_idx,
                "image": img_rgb,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "roll": float(row["roll"]),
                "pitch": float(row["pitch"]),
                "yaw": float(row["yaw"]),
            }
        )

    return records


def main():
    data = load_dataset(BASE_DIR)
    n = len(data)
    print(f"Loaded {n} frames")

    if n == 0:
        print("No data loaded, check paths.")
        return

    plt.ion()  # interactive mode
    fig, ax = plt.subplots(figsize=(7, 4))

    img_artist = ax.imshow(data[0]["image"])
    ax.axis("off")

    # text overlay in normalized (0–1) axes coordinates
    text_artist = ax.text(
        0.02,
        0.95,
        "",
        color="yellow",
        fontsize=10,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )

    for i in range(0, n, STEP):
        rec = data[i]

        # update image
        img_artist.set_data(rec["image"])

        # build text like:
        # p1  x y z
        # rpy roll pitch yaw
        txt = (
            f"frame: {rec['frame']}\n"
            f"p: {rec['x']:.3f}  {rec['y']:.3f}  {rec['z']:.3f}\n"
            f"rpy: {rec['roll']:.2f}  {rec['pitch']:.2f}  {rec['yaw']:.2f}"
        )
        text_artist.set_text(txt)

        ax.set_title("Image + GT (numbers)", fontsize=12)
        plt.draw()
        plt.pause(DELAY)

    print("Done – close the window to exit.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
