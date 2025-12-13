# Usage Guide

This guide covers setup, running the pose estimation pipeline, and Docker usage.

## Quick Start

```bash
# 1. Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python -m src.pipeline
```

## Project Structure

```
.
├── data/                   # Input data (images and ground truth)
│   ├── images/            # Image sequence (000000.png, 000015.png, ...)
│   └── camera_poses.txt   # Ground truth poses
├── results/               # Output files (created automatically)
│   ├── orientation_plot.html
│   ├── pose_comparison.mp4
│   └── evaluation_results.csv
├── src/                   # Source code
└── requirements.txt       # Python dependencies
```

## Setup

### Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- plotly >= 5.14.0
- matplotlib >= 3.7.0

## Running the Pipeline

### Default Run (Full Pipeline)

Runs evaluation with 3D plot and video generation:

```bash
python -m src.pipeline
```

This will:
- Process frames from `data/images/`
- Use ground truth from `data/camera_poses.txt`
- Generate outputs in `results/`

### Programmatic Usage

```python
from src.pipeline import PoseEstimationPipeline

# Initialize
pipeline = PoseEstimationPipeline(
    data_dir="data",
    results_dir="results",
    feature_method="ORB",      # or "SIFT"
    norm_type="Hamming",       # or "L2" for SIFT
    max_matches=500
)

# Setup components
pipeline.setup()

# Run with custom parameters
results = pipeline.run(
    step=15,                   # Frame interval
    create_plot=True,          # Generate 3D visualization
    create_video=True,         # Generate annotated video
    video_fps=10               # Video frame rate
)

# Access results
print(results['evaluation'])   # Error metrics
print(results['comparison_df'])  # Pandas DataFrame with GT vs EST
```

### Single Frame Pair Estimation

```python
from src.pipeline import PoseEstimationPipeline

pipeline = PoseEstimationPipeline()
pipeline.setup()

# Estimate pose between two frames
result = pipeline.run_single_pair(
    frame1_idx=0,
    frame2_idx=15,
    show_debug=True
)

print(f"Yaw: {result['yaw']:.2f}°")
print(f"Pitch: {result['pitch']:.2f}°")
print(f"Roll: {result['roll']:.2f}°")
```

### Custom Data Paths

```python
pipeline = PoseEstimationPipeline(
    data_dir="path/to/your/data",
    results_dir="path/to/output",
    gt_filename="camera_poses.txt"
)
```

## Using Individual Components

### Pose Estimation Only

```python
from src.core.camera_calibration import CameraCalibration
from src.core.pose_estimator import PoseEstimator
from src.utils.image_loader import load_image_pair

# Setup camera calibration
calibration = CameraCalibration()
K = calibration.get_matrix(image_width=1920, image_height=1080)

# Setup pose estimator
estimator = PoseEstimator(
    camera_matrix=K,
    feature_method="ORB",
    norm_type="Hamming",
    max_matches=500
)

# Estimate pose
img1, img2 = load_image_pair("data/images/000000.png",
                             "data/images/000015.png")
R, t = estimator.estimate(img1, img2)

print(f"Rotation matrix:\n{R}")
print(f"Translation direction: {t.flatten()}")
```

### Batch Processing

```python
from src.core.ground_truth_loader import GroundTruthLoader
from src.core.batch_processor import BatchProcessor

# Load ground truth
gt_loader = GroundTruthLoader("data/camera_poses.txt")
gt_loader.load()

# Process sequence
processor = BatchProcessor(
    images_dir="data/images",
    pose_estimator=estimator,
    ground_truth_loader=gt_loader
)

# Process at regular intervals
results = processor.process_at_interval(step=15)
# Or specific frames
results = processor.process_sequence([0, 15, 30, 45, 60])

print(f"Estimated yaw angles: {results['yaw']}")
print(f"Estimated pitch angles: {results['pitch']}")
print(f"Estimated roll angles: {results['roll']}")
```

### Evaluation and Visualization

```python
from src.core.pose_evaluator import PoseEvaluator
from src.core.visualizer import Visualizer

# Evaluate results
evaluator = PoseEvaluator(ground_truth_loader=gt_loader)
evaluation = evaluator.evaluate_sequence(results)
evaluator.print_summary(evaluation)

# Create visualizations
visualizer = Visualizer(output_dir="results")

# 3D trajectory plot
gt_trajectory = gt_loader.get_trajectory(step=1)
visualizer.plot_3d_trajectory(gt_trajectory, evaluation, step=15)

# Annotated video
visualizer.create_video(
    images_dir="data/images",
    evaluation_results=evaluation,
    output_filename="output.mp4",
    fps=10
)
```

## Docker Usage

### Build Docker Image

```bash
# Basic build
docker build -t pose-estimator:latest .

# Build with specific entry file
docker build --build-arg ENTRY_FILE=src/pipeline.py -t pose-estimator:latest .
```

### Run Interactive Container

**Linux/Mac (Bash):**
```bash
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  pose-estimator:latest
```

**Windows (PowerShell):**
```powershell
docker run --rm -it `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/results:/app/results `
  pose-estimator:latest
```

This opens an interactive Python shell with the project environment.

### Run Pipeline in Docker

**Linux/Mac (Bash):**
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  pose-estimator:latest \
  python -m src.pipeline
```

**Windows (PowerShell):**
```powershell
docker run --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/results:/app/results `
  pose-estimator:latest `
  python -m src.pipeline
```

### Open Bash Shell in Container

**Linux/Mac (Bash):**
```bash
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  --entrypoint /bin/bash \
  pose-estimator:latest
```

**Windows (PowerShell):**
```powershell
docker run --rm -it `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/results:/app/results `
  --entrypoint /bin/bash `
  pose-estimator:latest
```

### Docker with Custom Parameters

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  pose-estimator:latest \
  python -c "
from src.pipeline import PoseEstimationPipeline
pipeline = PoseEstimationPipeline()
pipeline.setup()
pipeline.run(step=30, create_plot=True, create_video=False)
"
```

## Utility Commands

### Clean Python Cache

```bash
# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove .pyc files
find . -type f -name "*.pyc" -delete

# Remove .pyo files
find . -type f -name "*.pyo" -delete

# Remove .egg-info directories
find . -type d -name "*.egg-info" -exec rm -rf {} +
```

On Windows:
```powershell
# PowerShell
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
```

### Remove Virtual Environment

```bash
rm -rf venv
```

On Windows:
```powershell
Remove-Item -Recurse -Force venv
```

## Standalone Scripts

The `src/scripts/` folder contains standalone visualization tools:

### Plot Trajectory from Angles

```bash
python src/scripts/plot_trajectory.py
```

Generates 3D trajectory visualization from roll/pitch/yaw angles in ground truth.

### Plot Position Vectors

```bash
python src/scripts/plot_vectors.py
```

Visualizes displacement vectors between camera positions.

## Troubleshooting

### Virtual Environment Issues

If you encounter `externally-managed-environment` error:
```bash
# Use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Import Errors

Ensure you're running from the project root:
```bash
cd /path/to/relative-pose-estimation
python -m src.pipeline
```

### Missing Data

Ensure your data directory structure matches:
```
data/
├── images/
│   ├── 000000.png
│   ├── 000015.png
│   └── ...
└── camera_poses.txt
```

### OpenCV Issues

If OpenCV fails to load:
```bash
# Install OpenGL libraries (Linux)
sudo apt-get install libgl1 libglib2.0-0

# Reinstall opencv-python
pip uninstall opencv-python
pip install opencv-python
```

## Performance Tips

### Faster Processing
- Use **ORB** instead of SIFT (faster, no licensing issues)
- Reduce `max_matches` parameter (default 500)
- Increase `step` for fewer frame pairs
- Reduce `nfeatures` in PoseEstimator

### Better Accuracy
- Use **SIFT** with `norm_type="L2"` (slower but more accurate)
- Increase `max_matches` (more correspondences)
- Decrease `step` for smaller inter-frame motion
- Increase `nfeatures` for more keypoints

Example:
```python
# Fast mode
pipeline = PoseEstimationPipeline(
    feature_method="ORB",
    max_matches=300
)
pipeline.run(step=30)

# Accurate mode
pipeline = PoseEstimationPipeline(
    feature_method="SIFT",
    norm_type="L2",
    max_matches=1000
)
pipeline.run(step=5)
```

## Next Steps

- See **CLAUDE.md** for detailed architecture and development patterns
- See **README.md** for project overview and mathematical background
- Check the `src/` directory for component-level documentation in docstrings
