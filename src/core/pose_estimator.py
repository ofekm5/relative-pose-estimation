"""
Pose estimation from image pairs using feature matching.
High-level component for computing relative camera pose (R, t) between two images.
"""

import numpy as np
import cv2


class PoseEstimator:
    """
    Estimates relative camera pose between two images using feature-based methods.

    Uses ORB or SIFT features, descriptor matching, Essential Matrix estimation,
    and pose recovery to compute rotation (R) and translation (t) between images.
    """

    def __init__(self,
                 camera_matrix,
                 feature_method="ORB",
                 norm_type="Hamming",
                 max_matches=500,
                 nfeatures=2000):
        """
        Initialize pose estimator.

        Args:
            camera_matrix: Camera intrinsic matrix K (3x3)
            feature_method: Feature detector method ('ORB' or 'SIFT')
            norm_type: Distance norm for matching ('Hamming' for ORB, 'L2' for SIFT)
            max_matches: Maximum number of matches to use for pose estimation
            nfeatures: Maximum number of features to extract (ORB only)
        """
        self.K = camera_matrix
        self.feature_method = feature_method
        self.norm_type = norm_type
        self.max_matches = max_matches
        self.nfeatures = nfeatures

        # Create feature extractor and matcher once
        self.extractor = self._create_feature_extractor()
        self.matcher = self._create_matcher()

    # ========================================
    # Internal Feature Extraction (merged from features.py)
    # ========================================

    def _create_feature_extractor(self):
        """
        Create feature detector+descriptor extractor.

        Returns:
            cv2 feature detector object
        """
        method = self.feature_method.upper()

        if method == "ORB":
            return cv2.ORB_create(nfeatures=self.nfeatures)

        if method == "SIFT":
            return cv2.SIFT_create()

        raise ValueError(f"Unknown feature extraction method: {method}")

    def _detect_and_compute(self, image):
        """
        Detect keypoints and compute descriptors for an image.

        Args:
            image: Input image (grayscale)

        Returns:
            tuple: (keypoints, descriptors)
        """
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)
        return keypoints, descriptors

    # ========================================
    # Internal Descriptor Matching (merged from matching.py)
    # ========================================

    def _create_matcher(self):
        """
        Create Brute-Force descriptor matcher.

        Returns:
            cv2.BFMatcher object
        """
        norm_type = self.norm_type.upper()

        if norm_type == "HAMMING":
            norm = cv2.NORM_HAMMING
        elif norm_type == "L2":
            norm = cv2.NORM_L2
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

        return cv2.BFMatcher(norm, crossCheck=True)

    def _match_descriptors(self, desc1, desc2):
        """
        Match descriptors between two images.

        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image

        Returns:
            list: Sorted matches (by distance), limited to max_matches
        """
        matches = self.matcher.match(desc1, desc2)

        # Sort by distance (best matches first)
        matches = sorted(matches, key=lambda m: m.distance)

        # Limit to max_matches
        if self.max_matches is not None:
            matches = matches[:self.max_matches]

        return matches

    # ========================================
    # Public API
    # ========================================

    def estimate(self, img1, img2):
        """
        Estimate relative pose between two images.

        Args:
            img1: First image (grayscale numpy array)
            img2: Second image (grayscale numpy array)

        Returns:
            tuple: (R, t) where:
                R: Rotation matrix (3x3) from camera1 to camera2
                t: Translation vector (3x1), direction only (unit scale)

        Raises:
            RuntimeError: If feature detection, matching, or pose estimation fails
        """
        # 1. Extract features
        kp1, desc1 = self._detect_and_compute(img1)
        kp2, desc2 = self._detect_and_compute(img2)

        if desc1 is None or desc2 is None:
            raise RuntimeError("Could not compute descriptors for one of the images.")

        # 2. Match descriptors
        matches = self._match_descriptors(desc1, desc2)

        if len(matches) < 5:
            raise RuntimeError(f"Insufficient matches: {len(matches)} (minimum 5 required)")

        # 3. Convert matches to point arrays
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

        # 4. Estimate Essential Matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            raise RuntimeError("Could not estimate Essential matrix.")

        # 5. Recover pose from Essential Matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        return R, t

    def estimate_with_debug(self, img1, img2):
        """
        Estimate relative pose with additional debugging information.

        Args:
            img1: First image (grayscale numpy array)
            img2: Second image (grayscale numpy array)

        Returns:
            dict: {
                'R': Rotation matrix (3x3),
                't': Translation vector (3x1),
                'num_matches': Number of matches used,
                'pts1': Matched points from img1,
                'pts2': Matched points from img2,
                'inliers': Number of inlier points after RANSAC
            }
        """
        # 1. Extract features
        kp1, desc1 = self._detect_and_compute(img1)
        kp2, desc2 = self._detect_and_compute(img2)

        if desc1 is None or desc2 is None:
            raise RuntimeError("Could not compute descriptors for one of the images.")

        # 2. Match descriptors
        matches = self._match_descriptors(desc1, desc2)

        if len(matches) < 5:
            raise RuntimeError(f"Insufficient matches: {len(matches)} (minimum 5 required)")

        # 3. Convert matches to point arrays
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

        # 4. Estimate Essential Matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            raise RuntimeError("Could not estimate Essential matrix.")

        # 5. Recover pose from Essential Matrix
        num_inliers, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        return {
            'R': R,
            't': t,
            'num_matches': len(matches),
            'pts1': pts1,
            'pts2': pts2,
            'inliers': num_inliers
        }
