# matcher.py
import cv2


def create_matcher(norm_type="Hamming", cross_check=True):
    """
    Create a Brute-Force matcher.
    norm_type: 'Hamming' for ORB, 'L2' for SIFT, etc.
    """
    norm_type = norm_type.upper()

    if norm_type == "HAMMING":
        norm = cv2.NORM_HAMMING
    elif norm_type == "L2":
        norm = cv2.NORM_L2
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

    bf = cv2.BFMatcher(norm, crossCheck=cross_check)
    return bf


def match_descriptors(desc1, desc2, matcher, sort_by_distance=True, max_matches=None):
    """
    Match descriptors between two images.
    Returns a list of cv2.DMatch.
    """
    matches = matcher.match(desc1, desc2)

    if sort_by_distance:
        matches = sorted(matches, key=lambda m: m.distance)

    if max_matches is not None:
        matches = matches[:max_matches]

    return matches
