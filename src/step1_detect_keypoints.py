import cv2
import numpy as np
from pathlib import Path

def load_gray_image(path: str) -> np.ndarray:
    """Load image from disk and convert to grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def detect_orb_keypoints_and_descriptors(
    img: np.ndarray,
    nfeatures: int = 3000
):
    """
    Detect ORB keypoints and descriptors for a single image.
    :param img: grayscale image (np.ndarray)
    :param nfeatures: max number of features to detect
    :return: (keypoints, descriptors)
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def draw_keypoints(img: np.ndarray, keypoints):
    """Return a BGR image with rich keypoints drawn on it."""
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = cv2.drawKeypoints(
        img_color,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return out


def main():
    # === 1. set your image paths here ===
    img1_path = "../images/img1.png"   # change to your first image path
    img2_path = "../images/img2.png"   # change to your second image path

    if not Path(img1_path).exists() or not Path(img2_path).exists():
        raise SystemExit("Update img1_path / img2_path to valid image files")

    # === 2. load images in grayscale ===
    img1 = load_gray_image(img1_path)
    img2 = load_gray_image(img2_path)

    # === 3. detect keypoints + descriptors ===
    kp1, des1 = detect_orb_keypoints_and_descriptors(img1, nfeatures=1500)
    kp2, des2 = detect_orb_keypoints_and_descriptors(img2, nfeatures=1500)

    print(f"Image 1: {len(kp1)} keypoints, descriptors shape: {des1.shape}")
    print(f"Image 2: {len(kp2)} keypoints, descriptors shape: {des2.shape}")

    # === 4. draw keypoints for visualization ===
    img1_kp = draw_keypoints(img1, kp1)
    img2_kp = draw_keypoints(img2, kp2)

    # === 5. show images (press any key to close) ===
    cv2.imshow("Image 1 - keypoints", img1_kp)
    cv2.imshow("Image 2 - keypoints", img2_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
