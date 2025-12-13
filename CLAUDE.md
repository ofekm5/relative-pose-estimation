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

**Run Full Pipeline:**
```bash
# From project root
python -m src.pipeline
```

**Or programmatically:**
```python
from src.pipeline import PoseEstimationPipeline

pipeline = PoseEstimationPipeline(
    data_dir="data",
    results_dir="results",
    feature_method="ORB",
    max_matches=500
)
pipeline.setup()

# Full evaluation with plot and video
results = pipeline.run(step=15, create_plot=True, create_video=True)

# Or single pair estimation
pipeline.run_single_pair(0, 15, show_debug=True)
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

See USAGE.md for setup and usage instructions.

### Dependencies

```bash
pip install opencv-python numpy pandas plotly matplotlib
```

## Architecture

### SOLID Architecture (Refactored)

The codebase follows SOLID principles with clear separation of concerns:

```
src/
├── pipeline.py                      # Root-level orchestrator
├── core/                           # High-level business logic
│   ├── camera_calibration.py      # Camera intrinsics management
│   ├── ground_truth_loader.py     # GT data loading and access
│   ├── pose_estimator.py          # Pose estimation (ORB/SIFT features + matching)
│   ├── batch_processor.py         # Multi-frame sequence processing
│   ├── pose_evaluator.py          # GT comparison and metrics
│   └── visualizer.py              # 3D plots and video generation
├── utils/                          # Low-level helper functions
│   ├── image_loader.py            # Image I/O utilities
│   └── geometry.py                # Rotation/geometry transformations
└── scripts/                        # Standalone visualization scripts
    ├── plot_trajectory.py
    └── plot_vectors.py
```

**Design Principles:**
- **Hierarchy:** Pipeline → Core (high-level steps) → Utils (low-level helpers)
- **Dependency Injection:** Components receive dependencies as parameters (e.g., camera_matrix, ground_truth_loader)
- **Single Responsibility:** Each class has one clear purpose
- **Testability:** Components are independent and can be tested in isolation

### Key Components

**Core Modules:**
- `pipeline.py` - Main orchestrator coordinating all components
- `camera_calibration.py` - Computes camera intrinsic matrix K from base parameters
- `ground_truth_loader.py` - Loads and provides access to GT poses
- `pose_estimator.py` - Estimates relative pose from image pairs (features + matching merged)
- `batch_processor.py` - Processes sequences of frames, accumulates trajectory
- `pose_evaluator.py` - Compares estimated vs GT poses, computes error metrics
- `visualizer.py` - Creates 3D plots and annotated videos

**Utils:**
- `image_loader.py` - Load single images or pairs with grayscale conversion
- `geometry.py` - Rotation matrix ↔ Euler angle conversions, error computation

### Data Flow

```
1. PoseEstimationPipeline.setup()
   ├── Load ground truth (GroundTruthLoader)
   ├── Compute camera matrix K (CameraCalibration)
   ├── Initialize pose estimator (PoseEstimator with K)
   ├── Setup batch processor (BatchProcessor)
   ├── Setup evaluator (PoseEvaluator)
   └── Setup visualizer (Visualizer)

2. PoseEstimationPipeline.run(step=15)
   ├── BatchProcessor.process_at_interval(step)
   │   ├── Load image pairs
   │   ├── PoseEstimator.estimate(img1, img2) → (R, t)
   │   ├── Transform to world frame using GT of first frame
   │   └── Accumulate trajectory
   ├── PoseEvaluator.evaluate_sequence(results)
   │   ├── Compare EST vs GT angles
   │   ├── Compute rotation/translation errors
   │   └── Generate statistics
   ├── Visualizer.plot_3d_trajectory()
   └── Visualizer.create_video()
```

### Camera Intrinsics

The camera matrix K is computed by scaling base intrinsics to match actual image resolution:

```python
from src.core.camera_calibration import CameraCalibration

calibration = CameraCalibration(
    fx_base=924.82939686,
    fy_base=920.4766382,
    cx_base=468.24930789,
    cy_base=353.65863024,
    base_width=960,
    base_height=720
)

K = calibration.get_matrix(image_width=1920, image_height=1080)
# Or directly from image
K = calibration.get_matrix_from_image(img)
```

**Key Parameters:**
- `fx, fy`: Focal lengths in pixels (scaled from base 924.8, 920.5)
- `cx, cy`: Principal point (scaled from base 468.2, 353.7)
- Base resolution: 960×720

## Coordinate System and Conventions

**Rotation Decomposition Convention (Y-up):**
- The function `rotation_to_euler_yup(R)` assumes: R = Ry(yaw) * Rx(pitch) * Rz(roll)
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
from src.core.ground_truth_loader import GroundTruthLoader

gt_loader = GroundTruthLoader("data/camera_poses.txt")
gt_loader.load()

# Get specific pose
pose = gt_loader.get_pose(frame_idx=15)
# Returns: {'frame': 15, 'x': ..., 'y': ..., 'z': ..., 'roll': ..., 'pitch': ..., 'yaw': ...}

# Get frame indices at intervals
frames = gt_loader.get_frame_indices(step=15)  # [0, 15, 30, ...]

# Get trajectory
positions = gt_loader.get_trajectory(step=15)  # (N, 3) array of [x, y, z]
```

### Data Location
- **Data directory:** `data/`
  - Ground truth: `data/camera_poses.txt`
  - Images: `data/images/XXXXXX.png`
  - Frame numbering: zero-padded 6 digits (e.g., 000000.png, 000015.png)
- **Results directory:** `results/`
  - Plots: `results/orientation_plot.html`
  - Videos: `results/pose_comparison.mp4`
  - CSV: `results/evaluation_results.csv`

## Important Implementation Details

### Rotation Matrix to Euler Angles
The decomposition handles gimbal lock singularities:
```python
from src.utils.geometry import rotation_to_euler_yup

yaw, pitch, roll = rotation_to_euler_yup(R)
# Returns angles in degrees
```

Implementation (in utils/geometry.py):
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
Available in `src.utils.geometry`:
- **rotation_error(R_est, R_gt):** Overall rotation error in degrees
- **translation_direction_error(t_est, t_gt):** Translation direction error (ignores scale)

Available in `src.core.pose_evaluator`:
- **Roll/Pitch/Yaw errors:** Individual angle errors in degrees
- **Summary statistics:** Mean, std, median, max, min for all metrics

## Common Development Patterns

### Using the Pipeline
```python
from src.pipeline import PoseEstimationPipeline

# Initialize
pipeline = PoseEstimationPipeline(
    data_dir="data",
    results_dir="results",
    feature_method="ORB",  # or "SIFT"
    norm_type="Hamming",   # or "L2" for SIFT
    max_matches=500
)

# Setup components
pipeline.setup()

# Run full evaluation
results = pipeline.run(
    step=15,
    create_plot=True,
    create_video=True,
    video_fps=10
)

# Access results
estimated = results['estimated']  # Frame-by-frame estimates
evaluation = results['evaluation']  # Error metrics
comparison_df = results['comparison_df']  # Pandas DataFrame
```

### Using Individual Components

**Pose Estimation Only:**
```python
from src.core.camera_calibration import CameraCalibration
from src.core.pose_estimator import PoseEstimator
from src.utils.image_loader import load_image_pair

# Setup
calibration = CameraCalibration()
K = calibration.get_matrix(1920, 1080)
estimator = PoseEstimator(camera_matrix=K, feature_method="ORB")

# Estimate pose
img1, img2 = load_image_pair("data/images/000000.png", "data/images/000015.png")
R, t = estimator.estimate(img1, img2)
```

**Batch Processing:**
```python
from src.core.batch_processor import BatchProcessor

processor = BatchProcessor(
    images_dir="data/images",
    pose_estimator=estimator,
    ground_truth_loader=gt_loader
)

# Process specific frames
results = processor.process_sequence([0, 15, 30, 45])
# Or at regular intervals
results = processor.process_at_interval(step=15)
```

### Adding a New Feature Extractor
1. Edit `src/core/pose_estimator.py`
2. Update `_create_feature_extractor()` method:
```python
if method == "NEW_METHOD":
    return cv2.SomeNewDetector_create()
```
3. Update `norm_type` parameter if needed (SIFT uses L2, ORB uses Hamming)
4. Test with `PoseEstimator(feature_method="NEW_METHOD")`

### Evaluating Different Frame Intervals
Modify `step` parameter:
```python
pipeline.run(step=30)  # Larger motion between frames
pipeline.run(step=5)   # Smaller motion
```

### Changing Camera Calibration
Update base intrinsics when creating CameraCalibration:
```python
calibration = CameraCalibration(
    fx_base=YOUR_FX,
    fy_base=YOUR_FY,
    cx_base=YOUR_CX,
    cy_base=YOUR_CY,
    base_width=YOUR_WIDTH,
    base_height=YOUR_HEIGHT
)
```

## File Organization

```
src/
  pipeline.py              # Main orchestrator
  core/
    camera_calibration.py  # Camera intrinsics
    ground_truth_loader.py # GT data management
    pose_estimator.py      # Pose estimation (ORB/SIFT + matching)
    batch_processor.py     # Multi-frame processing
    pose_evaluator.py      # Evaluation metrics
    visualizer.py          # Plots and videos
  utils/
    image_loader.py        # Image I/O
    geometry.py            # Rotation utilities
  scripts/
    plot_trajectory.py     # Standalone trajectory viz
    plot_vectors.py        # Standalone vector viz

data/                      # Input data
  camera_poses.txt         # Ground truth poses
  images/                  # PNG frames (000000.png, ...)

results/                   # Output directory
  orientation_plot.html    # 3D trajectory plot
  pose_comparison.mp4      # Annotated video
  evaluation_results.csv   # Error metrics CSV

include/                   # C++ headers (future)
CMakeLists.txt             # C++ build config
Dockerfile                 # Python Docker build
USAGE.md                   # Setup and usage guide
CLAUDE.md                  # AI assistant guidance
README.md                  # Project overview
```

## Known Limitations

- **Scale ambiguity:** Absolute translation magnitude cannot be recovered from monocular images
- **Rotation-translation coupling:** Large rotations may affect translation direction accuracy
- **Texture dependence:** Low-texture scenes produce fewer reliable matches
- **Motion assumptions:** Works best for small to moderate inter-frame motion
- **C++ implementation:** Architecture prepared (CMakeLists.txt, Dockerfile) but Python is the current production code

## Git Workflow

- Main branch: `main`
- Feature branches: `feat/*` (e.g., `feat/refactor-solid`)
- The `.gitignore` excludes Python cache files (`__pycache__/`, `*.pyc`) and virtual environments (`venv/`)
