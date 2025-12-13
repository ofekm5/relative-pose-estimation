# image_loader.py
import cv2


def load_image(path: str, to_gray: bool = True):
    """
    Load a single image from disk.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from: {path}")

    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def load_image_pair(path1: str, path2: str, to_gray: bool = True):
    """
    Load two images from disk and return them.
    """
    img1 = load_image(path1, to_gray=to_gray)
    img2 = load_image(path2, to_gray=to_gray)
    return img1, img2
