# feature_extractor.py
import cv2


def create_feature_extractor(method: str = "ORB"):
    """
    Create and return a feature detector+descriptor (e.g., ORB, SIFT).
    """
    method = method.upper()

    if method == "ORB":
        # ORB is free and usually enough
        return cv2.ORB_create(
                nfeatures=4000,
                scaleFactor=1.1,
                nlevels=12,
                fastThreshold=15,          # נסה 10–25
                scoreType=cv2.ORB_HARRIS_SCORE
)

    if method == "SIFT":
        # requires opencv-contrib-python and nonfree enabled
        return cv2.SIFT_create()

    raise ValueError(f"Unknown feature extraction method: {method}")


def detect_and_compute(image, extractor):
    """
    Detect keypoints and compute descriptors for a given image.
    Returns: keypoints, descriptors
    """
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors
