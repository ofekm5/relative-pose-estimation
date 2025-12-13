"""
Image loading utilities.
Low-level helper functions for loading images from disk.
"""

import cv2


def load_image(path, to_gray=True):
    """
    Load a single image from disk.

    Args:
        path: Path to image file
        to_gray: If True, convert to grayscale

    Returns:
        np.ndarray: Image as numpy array

    Raises:
        FileNotFoundError: If image cannot be read
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from: {path}")

    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def load_image_pair(path1, path2, to_gray=True):
    """
    Load two images from disk and return them.

    Args:
        path1: Path to first image
        path2: Path to second image
        to_gray: If True, convert to grayscale

    Returns:
        tuple: (img1, img2) as numpy arrays
    """
    img1 = load_image(path1, to_gray=to_gray)
    img2 = load_image(path2, to_gray=to_gray)
    return img1, img2
