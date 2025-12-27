# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# MANDATORY BEHAVIORAL REQUIREMENTS

## CRITICAL: Workflow Guidelines - NO EXCEPTIONS

**YOU MUST follow these rules before EVERY response without deviation:**

1. **Plan incrementally** - ALWAYS break complex tasks into small, trackable steps. NO monolithic approaches.
2. **Ask for clarification** - NEVER assume or guess missing requirements. STOP and ask if anything is unclear.
3. **Read selectively** - ONLY read explicitly needed files. NO broad directory scans or exploratory reading.
4. **Summarize concisely** - MAXIMUM 2-3 key findings and next steps. NO verbose explanations.

**VIOLATION OF THESE RULES IS UNACCEPTABLE.** If you find yourself about to violate these guidelines, STOP immediately and reconsider your approach.

## ABSOLUTE Response Style Requirements

**Default format for non-code answers (STRICTLY ENFORCED):**
- HARD LIMIT: 2-3 paragraphs maximum
- NEVER repeat previously established context
- ONLY provide actionable information
- NO unnecessary elaboration or preambles

**Code changes (MANDATORY):**
- Show ACTUAL file changes with diffs - descriptions are insufficient
- Explain rationale briefly (1-2 sentences max)
- Test ALL changes before claiming task completion
- NO untested code submissions

## NON-NEGOTIABLE Restrictions

**These restrictions override ANY other considerations, including helpfulness:**

- ❌ **NEVER create README files** unless user explicitly requests "create a README"
- ❌ **NEVER create documentation files** without explicit user request
- ✅ **ALWAYS prefer editing existing files** over creating new ones
- ✅ **PR titles/descriptions MUST be clear and concise** - no flowery language

**If you're tempted to violate these restrictions "because it would be helpful," STOP. These are non-negotiable requirements, not suggestions.**

## Enforcement Notice

These guidelines are **MANDATORY REQUIREMENTS**, not optional best practices. Any justification like "not always relevant" or "it would be more helpful to..." is a violation. When in conflict between these rules and your default behaviors, **THESE RULES WIN**.

## Project Overview

This is a **6-DoF (6 Degrees of Freedom) Relative Pose Estimation** system that uses classical computer vision techniques to estimate camera motion between two consecutive images. It uses **ORB features, feature matching, Essential Matrix estimation, and RecoverPose** to compute translation (Tx, Ty, Tz) and rotation (Roll, Pitch, Yaw) - no training or datasets required. **Optional VP (Vanishing Point) refinement** using LSD line detection can improve rotation estimation when applicable.

**Pipeline:** Image Loading → ORB Feature Extraction → Feature Matching → Essential Matrix → RecoverPose → (Optional VP Refinement) → 6-DoF Output (R, t, roll, pitch, yaw)

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
- `pipeline.py` - Main orchestrator coordinating all components (configured with nfeatures=4000, use_vp_refinement=True)
- `camera_calibration.py` - Computes camera intrinsic matrix K from base parameters
- `ground_truth_loader.py` - Loads and provides access to GT poses
- `pose_estimator.py` - Estimates relative pose from image pairs (ORB/SIFT features + matching + optional VP refinement using LSD line detection)
- `batch_processor.py` - Processes sequences of frames, accumulates trajectory (composes rotations: R_new = R_prev @ R_rel)
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
   │   ├── PoseEstimator.estimate(img1, img2, R_prev) → (R, t) [with optional VP refinement]
   │   ├── Transform to world frame: R_new_world = R_prev_world @ R_rel
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
- **ORB nfeatures:** 4000 keypoints maximum (increased for improved matching)
- **max_matches:** 500 best matches used for pose estimation
- **RANSAC threshold:** 1.0 pixel
- **RANSAC confidence:** 0.999
- **VP refinement:** Optional LSD-based refinement (enabled by default but applies only when reliability gates pass: vp_acc_min=8e5, vp_vp2_min=8e3)

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
estimator = PoseEstimator(
    camera_matrix=K,
    feature_method="ORB",
    nfeatures=4000,
    use_vp_refinement=True  # Optional VP refinement (applies when reliability gates pass)
)

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
- **VP refinement applicability:** VP refinement using LSD lines is enabled but only applies when strict reliability gates are met (vp_acc_min=8e5, vp_vp2_min=8e3). Many scenes (including simulator data) do not meet these thresholds, so refinement is rarely applied in practice.
- **Rotation composition:** CRITICAL - batch_processor.py composes rotations as R_new = R_prev @ R_rel (NOT R_prev @ R_rel.T). Incorrect composition causes yaw error accumulation over trajectories.
- **C++ implementation:** Architecture prepared (CMakeLists.txt, Dockerfile) but Python is the current production code

## Git Workflow

- Main branch: `main`
- Feature branches: `feat/*` (e.g., `feat/refactor-solid`)
- The `.gitignore` excludes Python cache files (`__pycache__/`, `*.pyc`) and virtual environments (`venv/`)
