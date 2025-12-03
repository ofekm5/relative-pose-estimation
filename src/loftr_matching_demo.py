import cv2
import torch
import kornia as K
import kornia.feature as KF
import numpy as np
from pathlib import Path


# ---- CONFIG ----
BASE_DIR = Path("../silmulator_data")
IMAGES_DIR = BASE_DIR / "images"
OUT_DIR = BASE_DIR / "loftr_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME1 = 0
FRAME2 = 90


def load_image_as_torch(path: Path, device: torch.device):
    """
    Load image as grayscale and convert to torch tensor of shape (1,1,H,W) in [0,1].
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    # K.image_to_tensor כבר מחזירה (1,1,H,W) לגרייסקייל
    t = K.image_to_tensor(img, keepdim=False).float() / 255.0  # (1,1,H,W)
    return t.to(device), img


def draw_matches(img0, img1, pts0, pts1, max_lines=300):
    """
    Simple visualization: stack images horizontally and draw lines between matches.
    img0, img1: numpy grayscale images
    pts0, pts1: Nx2 arrays of (x,y)
    """
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H = max(h0, h1)
    W = w0 + w1

    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[:h0, :w0, 0] = img0
    vis[:h0, :w0, 1] = img0
    vis[:h0, :w0, 2] = img0
    vis[:h1, w0:w0+w1, 0] = img1
    vis[:h1, w0:w0+w1, 1] = img1
    vis[:h1, w0:w0+w1, 2] = img1

    # normalize to color if needed
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # random subset if too many matches
    N = pts0.shape[0]
    idx = np.arange(N)
    if N > max_lines:
        idx = np.random.choice(N, size=max_lines, replace=False)

    for i in idx:
        x0, y0 = pts0[i]
        x1, y1 = pts1[i]
        pt1 = (int(x0), int(y0))
        pt2 = (int(x1) + w0, int(y1))

        cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(vis, pt1, 2, (255, 0, 0), -1)
        cv2.circle(vis, pt2, 2, (0, 0, 255), -1)

    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    img1_path = IMAGES_DIR / f"{FRAME1:06d}.png"
    img2_path = IMAGES_DIR / f"{FRAME2:06d}.png"

    print(f"[INFO] Image 1: {img1_path}")
    print(f"[INFO] Image 2: {img2_path}")

    # Load as torch tensors + original numpy for viz
    timg0, img0 = load_image_as_torch(img1_path, device)
    timg1, img1 = load_image_as_torch(img2_path, device)

    # Initialize LoFTR
    print("[INFO] Loading LoFTR (pretrained='outdoor')...")
    matcher = KF.LoFTR(pretrained="outdoor").to(device)
    matcher.eval()

    # Run matcher
    with torch.no_grad():
        input_batch = {
            "image0": timg0,  # (1,1,H,W)
            "image1": timg1   # (1,1,H,W)
        }
        output = matcher(input_batch)

    kpts0 = output["keypoints0"].cpu().numpy()  # Nx2
    kpts1 = output["keypoints1"].cpu().numpy()  # Nx2

    print(f"[INFO] LoFTR produced {kpts0.shape[0]} matches.")
    if kpts0.shape[0] == 0:
        print("[WARN] No matches found. Check images / model / device.")
        return

    # Print first few matches just to see numbers
    print("[INFO] First 5 matches (x0, y0) -> (x1, y1):")
    for i in range(min(5, kpts0.shape[0])):
        x0, y0 = kpts0[i]
        x1, y1 = kpts1[i]
        print(f"  ({x0:.1f}, {y0:.1f}) -> ({x1:.1f}, {y1:.1f})")

    # Create visualization and save
    vis = draw_matches(img0, img1, kpts0, kpts1, max_lines=300)
    out_path = OUT_DIR / f"matches_{FRAME1:06d}_{FRAME2:06d}.png"
    cv2.imwrite(str(out_path), vis)
    print(f"[INFO] Saved matches visualization to: {out_path}")


if __name__ == "__main__":
    main()
