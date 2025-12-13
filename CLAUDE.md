# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Before major changes
1. Use a planner, provide step-by-step descriptions
2. Make small changes each step
3. Ask questions if data missing or if there are unclear instructions

## Project Overview

This is a **6-DoF (6 Degrees of Freedom) Relative Pose Estimation** system that uses classical computer vision techniques to estimate camera motion between two consecutive images. It uses **ORB features, feature matching, Essential Matrix estimation, and RecoverPose** to compute translation (Tx, Ty, Tz) and rotation (Roll, Pitch, Yaw) - no training or datasets required.

**Pipeline:** Image Loading → ORB Feature Extraction → Feature Matching → Essential Matrix → RecoverPose → 6-DoF Output (R, t, roll, pitch, yaw)

**Language:** Python 3 with OpenCV (C++ architecture prepared but not yet implemented)

## Running the Code

### Python Evaluation (Current Implementation)

**Single Frame Pair:**
```bash
# From project root
python -c "
from src.pose_matcher import PoseMatcher
from src.image_loader import load_image_pair

matcher = PoseMatcher('silmulator_data/simple_movement', 'silmulator_data/simple_movement/camera_poses.txt')
img1, img2 = load_image_pair('silmulator_data/simple_movement/images/000000.png',
                              'silmulator_data/simple_movement/images/000015.png', to_gray=True)
yaw, pitch, roll = matcher.match(img1, img2, prev_frame_index=0)
print(f'Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°')
"
```

**Multi-Frame Evaluation with Visualization:**
```bash
# From project root
python -c "
from src.plots_graths import PosePlotter

plotter = PosePlotter('silmulator_data/simple_movement', step=15)
plotter.run()  # Generates 3D interactive plot
plotter.make_video('output.mp4', fps=10)  # Generates MP4 with GT vs Estimated overlay
"
```

### C++ Build (Future)

```bash
# Local build
make build
make run IMG1=path/to/img1.jpg IMG2=path/to/img2.jpg

# Docker build
make docker-build
make docker-run
```

See BUILD.md for comprehensive build documentation.

### Dependencies

```bash
pip install opencv-python numpy pandas plotly matplotlib
```

## Architecture

### Module Structure

The codebase is organized into independent, testable modules:

**Core Modules:**
- `image_loader.py` - Load single or pairs of images, optionally convert to grayscale
- `feature_extractor.py` - Create ORB/SIFT detectors and compute keypoints + descriptors
- `matcher.py` - Match descriptors between images using BFMatcher with configurable distance metrics
- `gt_utils.py` - Load and parse ground truth poses from simulator, compute GT deltas
- `pose_matcher.py` - Main pipeline integrating all modules + pose computation
- `plots_graths.py` - 3D visualization and video generation
- `plot_from_yaw_pitch_roll.py` - Standalone trajectory visualization
- `toy_3d_vectors.py` - Simple path visualization utility

**Key Classes:**
- `PoseMatcher` (pose_matcher.py) - High-level API for pose estimation
- `PosePlotter` (plots_graths.py) - Visualization and video generation

### Data Flow

```
1. Load image pair (image_loader.py)
2. Extract ORB features (feature_extractor.py)
3. Match features (matcher.py)
4. Convert matches to point arrays
5. Estimate Essential Matrix: cv2.findEssentialMat(pts1, pts2, K)
6. Recover pose: cv2.recoverPose(E, pts1, pts2, K)
7. Convert R to Euler angles (rotmat_to_ypr_y_up)
8. Compare against GT (gt_utils.py)
```

### Camera Intrinsics

The camera matrix K is computed by scaling base intrinsics to match actual image resolution:

```python
h, w = img1.shape[:2]
scale_x = w / 960.0
scale_y = h / 720.0
fx = 924.82939686 * scale_x
fy = 920.4766382 * scale_y
cx = 468.24930789 * scale_x
cy = 353.65863024 * scale_y

K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
])
```

**Key Parameters:**
- `fx, fy`: Focal lengths in pixels (scaled from base 924.8, 920.5)
- `cx, cy`: Principal point (scaled from base 468.2, 353.7)
- Base resolution: 960×720

## Coordinate System and Conventions

**Rotation Decomposition Convention (Y-up):**
- The function `rotmat_to_ypr_y_up(R)` assumes: R = Ry(yaw) * Rx(pitch) * Rz(roll)
- Yaw rotates around +Y axis (vertical)
- Pitch rotates around +X axis
- Roll rotates around +Z axis
- This matches simulator's coordinate frame

**Essential Matrix Properties:**
- Translation T is recovered **up to scale** (direction only, not magnitude)
- Rotation R is exact regardless of scale
- Minimum 5 point correspondences required
- RANSAC automatically filters outliers

## Ground Truth Data

### File Format: `camera_poses.txt`
```
frame  x  y  z  roll (deg)  pitch (deg)  yaw (deg)
0      ...
15     ...
```

**Loading GT:**
```python
from gt_utils import load_gt_poses
df_gt = load_gt_poses("silmulator_data/simple_movement/camera_poses.txt")
```

**Computing GT Deltas:**
```python
from gt_utils import evaluate_pair
pos_err, d_roll, d_pitch, d_yaw, t_est_cam1 = evaluate_pair(base_dir, df_gt, frame1, frame2)
```

### Data Location
- Ground truth: `silmulator_data/simple_movement/camera_poses.txt`
- Images: `silmulator_data/simple_movement/images/XXXXXX.png`
- Frame numbering: zero-padded 6 digits (e.g., 000330.png)

## Important Implementation Details

### Rotation Matrix to Euler Angles
The decomposition handles gimbal lock singularities:
```python
pitch = -np.arcsin(R[1, 2])
if np.isclose(np.cos(pitch), 0.0, atol=1e-6):
    # Gimbal lock case
    roll = 0.0
    yaw = np.arctan2(-R[0, 1], R[0, 0])
else:
    # Normal case
    yaw = np.arctan2(R[0, 2], R[2, 2])
    roll = np.arctan2(R[1, 0], R[1, 1])
```

### Feature Matching Parameters
- **ORB nfeatures:** 2000 keypoints maximum
- **max_matches:** 500 best matches used for pose estimation
- **RANSAC threshold:** 1.0 pixel
- **RANSAC confidence:** 0.999

### Error Metrics
- **Position error:** Euclidean norm of translation difference `||t_est - t_gt||`
- **Angle errors:** Absolute differences `|angle_est - angle_gt|` in degrees
- Note: Translation comparison requires scale alignment

## Common Development Patterns

### Using PoseMatcher for Evaluation
```python
from src.pose_matcher import PoseMatcher
from src.image_loader import load_image_pair

# Initialize with ground truth data
matcher = PoseMatcher(
    base_dir='silmulator_data/simple_movement',
    gt_path='silmulator_data/simple_movement/camera_poses.txt',
    feature_method="ORB",  # or "SIFT"
    norm_type="Hamming",   # or "L2" for SIFT
    max_matches=500
)

# Load images
img1, img2 = load_image_pair(img1_path, img2_path, to_gray=True)

# Estimate pose
yaw, pitch, roll = matcher.match(img1, img2, prev_frame_index=frame1_idx)
```

### Adding a New Feature Extractor
1. Add method to `create_feature_extractor()` in `feature_extractor.py`
2. Update `norm_type` parameter in matcher (SIFT uses L2, ORB uses Hamming)
3. Test with `PoseMatcher(feature_method="NEW_METHOD")`

### Evaluating Different Frame Intervals
Modify `step` variable when creating PosePlotter:
```python
plotter = PosePlotter(base_path, step=15)  # Increase for larger motion
```

### Changing Camera Calibration
Update base intrinsics in `pose_matcher.py` `_build_K()` method if using different camera.

## File Organization

```
src/
  image_loader.py       # Image I/O utilities
  feature_extractor.py  # ORB/SIFT feature detection
  matcher.py            # Descriptor matching (BFMatcher)
  gt_utils.py           # Ground truth loading and parsing
  pose_matcher.py       # Main pipeline + rotation conversion
  plots_graths.py       # 3D visualization and video generation
  plot_from_yaw_pitch_roll.py  # Trajectory visualization
  toy_3d_vectors.py     # Simple visualization utility

silmulator_data/simple_movement/
  camera_poses.txt      # Ground truth poses (frame, x, y, z, roll, pitch, yaw)
  images/               # PNG frames from simulator (000000.png, 000001.png, ...)
  results/              # Output directory for plots and videos

include/                # C++ headers (prepared, currently empty)
CMakeLists.txt          # C++ build configuration
Dockerfile              # Multi-stage Docker build
Makefile                # Build orchestration wrapper
BUILD.md                # Comprehensive build documentation
README.md               # Project overview and mathematical background
```

## Known Limitations

- **Scale ambiguity:** Absolute translation magnitude cannot be recovered from monocular images
- **Rotation-translation coupling:** Large rotations may affect translation direction accuracy
- **Texture dependence:** Low-texture scenes produce fewer reliable matches
- **Motion assumptions:** Works best for small to moderate inter-frame motion
- **Comments in Hebrew:** Some code comments are in Hebrew (primarily in gt_utils.py)
- **C++ implementation:** Architecture prepared (CMakeLists.txt, Dockerfile) but Python is the current production code

## Git Workflow

- Main branch: `main`
- Feature branches: `feat/*` (e.g., `feat/dockerfile`)
- The `.gitignore` excludes Python cache files (`__pycache__/`, `*.pyc`)
