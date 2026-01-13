"""
Pose estimation from image pairs using feature matching.
High-level component for computing relative camera pose (R, t) between two images.
"""

import numpy as np
import cv2
import itertools


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
                 nfeatures=4000,
                 use_vp_refinement=False,
                 vp_max_lines=120,
                 vp_max_pairs=3000,
                 vp_acc_min=8e5,
                 vp_vp2_min=8000.0,
                 vp_iters=12,
                 vp_lm_lambda=1e-2,
                 vp_cost_improve_eps=1e-3):
        """
        Initialize pose estimator.

        Args:
            camera_matrix: Camera intrinsic matrix K (3x3)
            feature_method: Feature detector method ('ORB' or 'SIFT')
            norm_type: Distance norm for matching ('Hamming' for ORB, 'L2' for SIFT)
            max_matches: Maximum number of matches to use for pose estimation
            nfeatures: Maximum number of features to extract (ORB only)
            use_vp_refinement: Enable vanishing point refinement using line segments
            vp_max_lines: Maximum LSD lines to use for VP extraction
            vp_max_pairs: Maximum line pairs for VP voting
            vp_acc_min: Minimum accumulator value for reliable VP
            vp_vp2_min: Minimum VP2 score for reliability
            vp_iters: Number of iterations for SO(3) optimization
            vp_lm_lambda: Levenberg-Marquardt damping factor
            vp_cost_improve_eps: Minimum cost improvement to accept VP refinement
        """
        self.K = camera_matrix
        self.feature_method = feature_method
        self.norm_type = norm_type
        self.max_matches = max_matches
        self.nfeatures = nfeatures

        # VP refinement parameters
        self.use_vp_refinement = use_vp_refinement
        self.vp_max_lines = vp_max_lines
        self.vp_max_pairs = vp_max_pairs
        self.vp_acc_min = vp_acc_min
        self.vp_vp2_min = vp_vp2_min
        self.vp_iters = vp_iters
        self.vp_lm_lambda = vp_lm_lambda
        self.vp_cost_improve_eps = vp_cost_improve_eps

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
            return cv2.ORB_create(
                nfeatures=self.nfeatures,
                scaleFactor=1.1,
                nlevels=12,
                fastThreshold=15,
                scoreType=cv2.ORB_HARRIS_SCORE
            )

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
    # Vanishing Point (VP) Refinement
    # ========================================

    @staticmethod
    def _detect_lsd_lines(gray):
        """
        Detect line segments using LSD (Line Segment Detector).

        Args:
            gray: Grayscale image

        Returns:
            np.ndarray: Array of line segments (Nx4), each row is [x1, y1, x2, y2]
        """
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines, _, _, _ = lsd.detect(gray)
        if lines is None:
            return np.zeros((0, 4), dtype=np.float64)
        return lines.reshape(-1, 4).astype(np.float64)

    @staticmethod
    def _line_angle_and_len(line):
        """
        Compute angle and length of a line segment.

        Args:
            line: Line segment [x1, y1, x2, y2]

        Returns:
            tuple: (angle_rad, length)
        """
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy) + 1e-9)
        angle = float(np.arctan2(dy, dx))
        return angle, length

    @staticmethod
    def _hom_line_from_segment(line):
        """
        Convert line segment to homogeneous line representation.

        Args:
            line: Line segment [x1, y1, x2, y2]

        Returns:
            np.ndarray: Normalized homogeneous line [a, b, c]
        """
        x1, y1, x2, y2 = line
        p1 = np.array([x1, y1, 1.0], dtype=np.float64)
        p2 = np.array([x2, y2, 1.0], dtype=np.float64)
        l = np.cross(p1, p2)
        n = np.linalg.norm(l[:2]) + 1e-12
        return l / n

    @staticmethod
    def _vp_dir_from_image_point(vp_xy, K):
        """
        Convert vanishing point in image to 3D direction in camera frame.

        Args:
            vp_xy: Vanishing point coordinates (x, y)
            K: Camera intrinsic matrix

        Returns:
            np.ndarray: Normalized 3D direction vector
        """
        x, y = float(vp_xy[0]), float(vp_xy[1])
        v = np.array([x, y, 1.0], dtype=np.float64)
        d = np.linalg.inv(K) @ v
        d = d / (np.linalg.norm(d) + 1e-12)
        # Half-sphere convention (z > 0)
        if d[2] < 0:
            d = -d
        return d

    @staticmethod
    def _dir_to_grid_idx(d, n_lat=90, n_lon=360):
        """
        Convert 3D direction to polar grid indices.

        Args:
            d: 3D direction vector
            n_lat: Number of latitude bins (0 to 90 degrees)
            n_lon: Number of longitude bins (0 to 360 degrees)

        Returns:
            tuple: (lat_idx, lon_idx)
        """
        # Half-sphere: z > 0
        lat = np.arctan2(np.hypot(d[0], d[1]), d[2])  # 0..pi/2
        lon = np.arctan2(d[1], d[0])                  # -pi..pi
        lat_deg = np.rad2deg(lat)
        lon_deg = (np.rad2deg(lon) + 360.0) % 360.0
        lat_i = int(np.clip(lat_deg, 0, n_lat - 1))
        lon_i = int(np.clip(lon_deg, 0, n_lon - 1))
        return lat_i, lon_i

    def _estimate_manhattan_dirs(self, gray, n_lat=90, n_lon=360, rng_seed=0):
        """
        Extract 3 orthogonal Manhattan directions from image using VP extraction.

        Uses LSD line detection + polar grid voting on Gaussian sphere with
        weight = |l1||l2|sin(2θ) as in VP-SLAM (arXiv:2210.12756).

        Args:
            gray: Grayscale image
            n_lat: Number of latitude bins for polar grid
            n_lon: Number of longitude bins for polar grid
            rng_seed: Random seed for line pair sampling

        Returns:
            tuple: (Delta, ok, debug_info) where:
                Delta: (3,3) matrix with Manhattan directions as columns
                ok: Boolean indicating success
                debug_info: Dict with extraction statistics
        """
        lines = self._detect_lsd_lines(gray)
        dbg = {"num_lines": int(lines.shape[0])}

        if lines.shape[0] < 10:
            return None, False, dbg

        # Sort lines by length, keep top max_lines
        lens_all = np.array([self._line_angle_and_len(l)[1] for l in lines])
        idx = np.argsort(-lens_all)[:min(self.vp_max_lines, len(lines))]
        lines = lines[idx]
        lens = lens_all[idx]

        # Convert to homogeneous lines and get angles
        hlines = np.array([self._hom_line_from_segment(l) for l in lines])
        angles = np.array([self._line_angle_and_len(l)[0] for l in lines])

        # Polar grid accumulator
        acc = np.zeros((n_lat, n_lon), dtype=np.float64)

        # Generate line pairs for VP voting
        m = len(lines)
        total_pairs = m * (m - 1) // 2
        if total_pairs <= self.vp_max_pairs:
            pairs = list(itertools.combinations(range(m), 2))
        else:
            # Random sampling if too many pairs
            rng = np.random.default_rng(rng_seed)
            pairs = []
            for _ in range(self.vp_max_pairs):
                i = int(rng.integers(0, m))
                j = int(rng.integers(0, m))
                if i != j:
                    if i > j:
                        i, j = j, i
                    pairs.append((i, j))

        # Vote for vanishing points
        for i, j in pairs:
            li = hlines[i]
            lj = hlines[j]
            vp = np.cross(li, lj)
            if abs(vp[2]) < 1e-9:
                continue
            vp_xy = (vp[0] / vp[2], vp[1] / vp[2])

            # Angle between lines
            theta = abs(angles[i] - angles[j])
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            theta = abs(theta)

            # Weight: |l1||l2|sin(2θ)
            w = float(lens[i] * lens[j] * abs(np.sin(2.0 * theta)))
            if w <= 0:
                continue

            # Convert VP to 3D direction and vote
            d = self._vp_dir_from_image_point(vp_xy, self.K)
            lat_i, lon_i = self._dir_to_grid_idx(d, n_lat=n_lat, n_lon=n_lon)
            acc[lat_i, lon_i] += w

        acc_max = float(np.max(acc))
        dbg["acc_max"] = acc_max
        dbg["lines_used"] = int(m)

        if acc_max <= 0:
            return None, False, dbg

        # VP1: Find strongest vanishing point
        lat1, lon1 = np.unravel_index(np.argmax(acc), acc.shape)
        lat1_rad = np.deg2rad(lat1 + 0.5)
        lon1_rad = np.deg2rad(lon1 + 0.5)
        v1 = np.array([
            np.sin(lat1_rad) * np.cos(lon1_rad),
            np.sin(lat1_rad) * np.sin(lon1_rad),
            np.cos(lat1_rad)
        ], dtype=np.float64)
        v1 /= (np.linalg.norm(v1) + 1e-12)

        # VP2: Find second VP orthogonal to v1 (on great circle)
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, v1)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        a = np.cross(v1, tmp)
        a /= (np.linalg.norm(a) + 1e-12)
        b = np.cross(v1, a)
        b /= (np.linalg.norm(b) + 1e-12)

        best_score = -1.0
        v2 = None
        for deg in range(360):
            ang = np.deg2rad(deg)
            cand = np.cos(ang) * a + np.sin(ang) * b
            cand /= (np.linalg.norm(cand) + 1e-12)
            lat_i, lon_i = self._dir_to_grid_idx(cand, n_lat=n_lat, n_lon=n_lon)
            s = acc[lat_i, lon_i]
            if s > best_score:
                best_score = float(s)
                v2 = cand

        dbg["vp2_score"] = float(best_score)

        if v2 is None or best_score <= 0:
            return None, False, dbg

        # VP3: Cross product to ensure orthogonality
        v3 = np.cross(v1, v2)
        v3 /= (np.linalg.norm(v3) + 1e-12)
        v2 = np.cross(v3, v1)
        v2 /= (np.linalg.norm(v2) + 1e-12)

        # Delta: columns are Manhattan directions in camera frame
        Delta = np.stack([v1, v2, v3], axis=1)
        return Delta, True, dbg

    @staticmethod
    def _so3_exp(w):
        """
        Exponential map from so(3) to SO(3) using Rodrigues formula.

        Args:
            w: Rotation vector (3,)

        Returns:
            np.ndarray: Rotation matrix (3x3)
        """
        R, _ = cv2.Rodrigues(w.reshape(3, 1))
        return R

    @staticmethod
    def _vp_cost(R_iw, Delta_cam, D_world):
        """
        Compute VP alignment cost: sum of angular errors between detected VPs
        and expected Manhattan directions.

        Cost = sum_k arccos(δ_k · (R d_k))

        Args:
            R_iw: Current rotation estimate (3x3)
            Delta_cam: Detected Manhattan directions in camera frame (3x3)
            D_world: Expected Manhattan directions in world frame (3x3)

        Returns:
            float: Total angular error in radians
        """
        cost = 0.0
        for k in range(3):
            delta = Delta_cam[:, k]
            d = D_world[:, k]
            u = R_iw @ d
            s = float(np.clip(delta @ u, -1.0, 1.0))
            cost += float(np.arccos(s))
        return cost

    def _optimize_rotation_from_vps(self, R_init, Delta_cam, D_world):
        """
        Optimize rotation matrix to align detected VPs with world Manhattan directions.

        Uses Levenberg-Marquardt on SO(3) with cost E(R) = sum_k arccos(δ_k · (R d_k)).

        Args:
            R_init: Initial rotation estimate (3x3)
            Delta_cam: Detected Manhattan directions in camera frame (3x3)
            D_world: Expected Manhattan directions in world frame (3x3)

        Returns:
            np.ndarray: Optimized rotation matrix (3x3)
        """
        R = R_init.copy()

        for _ in range(self.vp_iters):
            r_list = []
            J_list = []

            for k in range(3):
                delta = Delta_cam[:, k]
                d = D_world[:, k]
                u = R @ d

                s = float(np.clip(delta @ u, -1.0, 1.0))
                e = float(np.arccos(s))
                r_list.append(e)

                # Jacobian: d(arccos(δ·(Rd)))/dw = -(1/sqrt(1-s²)) * (δ × (Rd))
                denom = np.sqrt(max(1e-12, 1.0 - s * s))
                cross = np.cross(delta, u)
                J = -(1.0 / denom) * cross
                J_list.append(J.reshape(1, 3))

            r = np.array(r_list, dtype=np.float64).reshape(3, 1)
            J = np.vstack(J_list)

            # Levenberg-Marquardt: (J^T J + λI) dw = -J^T r
            H = J.T @ J + self.vp_lm_lambda * np.eye(3)
            g = J.T @ r

            try:
                dw = -np.linalg.solve(H, g).reshape(3,)
            except np.linalg.LinAlgError:
                break

            # Update rotation: R ← exp(dw) * R
            R = self._so3_exp(dw) @ R

            # Convergence check
            if np.linalg.norm(dw) < 1e-7:
                break

        return R

    # ========================================
    # Public API
    # ========================================

    def estimate(self, img1, img2, R_prev=None):
        """
        Estimate relative pose between two images.

        Args:
            img1: First image (grayscale numpy array)
            img2: Second image (grayscale numpy array)
            R_prev: Optional absolute rotation of img1 (3x3). Required for VP refinement.

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
        _, R_rel, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        # 6. VP refinement (if enabled and R_prev provided)
        if self.use_vp_refinement and R_prev is not None:
            # Compute initial absolute rotation for img2
            R_new_init = R_prev @ R_rel

            # Extract VPs from both images
            Delta_prev, ok1, dbg1 = self._estimate_manhattan_dirs(img1, rng_seed=0)
            Delta_new, ok2, dbg2 = self._estimate_manhattan_dirs(img2, rng_seed=1)

            # Check reliability gates
            vp_good_prev = ok1 and (dbg1.get("acc_max", 0.0) >= self.vp_acc_min) and \
                          (dbg1.get("vp2_score", 0.0) >= self.vp_vp2_min)
            vp_good_new = ok2 and (dbg2.get("acc_max", 0.0) >= self.vp_acc_min) and \
                         (dbg2.get("vp2_score", 0.0) >= self.vp_vp2_min)

            if vp_good_prev and vp_good_new:
                # Build world Manhattan directions from prev frame
                # δ ~ R d  =>  d = R^T δ
                D_world = R_prev.T @ Delta_prev

                # Compute initial cost
                cost_init = self._vp_cost(R_new_init, Delta_new, D_world)

                # Optimize rotation
                R_opt = self._optimize_rotation_from_vps(R_new_init, Delta_new, D_world)

                # Compute optimized cost
                cost_opt = self._vp_cost(R_opt, Delta_new, D_world)

                # Accept only if cost improves
                if cost_opt < cost_init - self.vp_cost_improve_eps:
                    # Update R_rel to reflect the refined rotation
                    R_rel = R_prev.T @ R_opt

        return R_rel, t

    def estimate_with_debug(self, img1, img2, R_prev=None):
        """
        Estimate relative pose with additional debugging information.

        Args:
            img1: First image (grayscale numpy array)
            img2: Second image (grayscale numpy array)
            R_prev: Optional absolute rotation of img1 (3x3). Required for VP refinement.

        Returns:
            dict: {
                'R': Rotation matrix (3x3),
                't': Translation vector (3x1),
                'num_matches': Number of matches used,
                'pts1': Matched points from img1,
                'pts2': Matched points from img2,
                'inliers': Number of inlier points after RANSAC,
                'vp_used': Boolean indicating if VP refinement was applied,
                'vp_debug': Dict with VP extraction statistics (if VP used)
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
        num_inliers, R_rel, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        # Initialize debug info
        debug_info = {
            'R': R_rel,
            't': t,
            'num_matches': len(matches),
            'pts1': pts1,
            'pts2': pts2,
            'inliers': num_inliers,
            'vp_used': False,
            'vp_debug': {}
        }

        # 6. VP refinement (if enabled and R_prev provided)
        if self.use_vp_refinement and R_prev is not None:
            # Compute initial absolute rotation for img2
            R_new_init = R_prev @ R_rel

            # Extract VPs from both images
            Delta_prev, ok1, dbg1 = self._estimate_manhattan_dirs(img1, rng_seed=0)
            Delta_new, ok2, dbg2 = self._estimate_manhattan_dirs(img2, rng_seed=1)

            # Store VP debug info
            debug_info['vp_debug'] = {
                'prev_frame': dbg1,
                'new_frame': dbg2,
                'vp_extracted': ok1 and ok2,
            }

            # Check reliability gates
            vp_good_prev = ok1 and (dbg1.get("acc_max", 0.0) >= self.vp_acc_min) and \
                          (dbg1.get("vp2_score", 0.0) >= self.vp_vp2_min)
            vp_good_new = ok2 and (dbg2.get("acc_max", 0.0) >= self.vp_acc_min) and \
                         (dbg2.get("vp2_score", 0.0) >= self.vp_vp2_min)

            debug_info['vp_debug']['reliability'] = {
                'prev_reliable': vp_good_prev,
                'new_reliable': vp_good_new
            }

            if vp_good_prev and vp_good_new:
                # Build world Manhattan directions from prev frame
                D_world = R_prev.T @ Delta_prev

                # Compute initial cost
                cost_init = self._vp_cost(R_new_init, Delta_new, D_world)

                # Optimize rotation
                R_opt = self._optimize_rotation_from_vps(R_new_init, Delta_new, D_world)

                # Compute optimized cost
                cost_opt = self._vp_cost(R_opt, Delta_new, D_world)

                debug_info['vp_debug']['optimization'] = {
                    'cost_init': cost_init,
                    'cost_opt': cost_opt,
                    'cost_improved': cost_opt < cost_init - self.vp_cost_improve_eps
                }

                # Accept only if cost improves
                if cost_opt < cost_init - self.vp_cost_improve_eps:
                    # Update R_rel to reflect the refined rotation
                    R_rel = R_prev.T @ R_opt
                    debug_info['R'] = R_rel
                    debug_info['vp_used'] = True

        return debug_info
