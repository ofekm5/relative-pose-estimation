# relative-pose-estimation

### **Classical Multi-View Geometry — No Training Required**

This repository implements a complete system for estimating **6 Degrees of Freedom (6-DoF)** camera motion between two images taken consecutively by a moving robot (e.g., drone, car, rover).

The method uses **ORB features**, **feature matching**, **Essential Matrix estimation**, and **RecoverPose** to compute:

* **Translation:** Tx, Ty, Tz
* **Rotation:** Roll, Pitch, Yaw

This pipeline requires **no dataset** and **no learning**, and works entirely with classical geometric computer-vision methods implemented in OpenCV.

---


## Simulator Mapping (Linux 2024 Support)

We used the **rbdlabhaifa** simulator to build **ground truth (GT)**.


### Original Repository
https://github.com/rbdlabhaifa/simulatorMapping

### Linux 2024 Support (Updated Version)
We modified the simulator to also support **Linux (2024)**.

Updated version:
https://github.com/orki360/simulatorMapping


## 📦 Project Structure

```
relative-pose-estimation/
├── src/
│   ├── pipeline.py              # Main orchestrator
│   ├── run_phone_data.py        # Phone data runner
│   ├── run_simulator_data.py    # Simulator data runner
│   ├── run_vo_database_salah.py # Salah Qadah dataset runner
│   ├── run_single_pair.py       # Single pair estimator
│   ├── core/                    # High-level components
│   │   ├── camera_calibration.py
│   │   ├── ground_truth_loader.py
│   │   ├── pose_estimator.py
│   │   ├── batch_processor.py
│   │   ├── pose_evaluator.py
│   │   └── visualizer.py
│   └── utils/                   # Helper functions
│       ├── image_loader.py
│       └── geometry.py
├── evaluation-runs/             # Evaluation datasets
│   ├── phone-data/
│   │   ├── data/                # Images, ground truth, calibration
│   │   └── results/             # Output files
│   ├── vo_dataset_salah/
│   │   ├── data/                # Images, ground truth, calibration
│   │   └── results/             # Output files
│   ├── simulator-data/
│   │   ├── data/                # Images, ground truth
│   │   └── results/             # Output files
│   └── single-pair/
│       └── images/              # Two test images
├── Dockerfile
└── README.md
```

---

## 🧠 Method Overview

The 6-DoF relative motion is computed using the following pipeline:

```
ORB → Feature Matching → Essential Matrix → RecoverPose → 6-DoF Output
```

### **1. ORB Feature Extraction**

Detect repeatable keypoints and compute compact binary descriptors that uniquely represent local visual patterns.

### **2. Feature Matching**

Match descriptors between the two images using:

* **BFMatcher (Hamming distance)** or
* **FLANN (LSH)**

This produces corresponding pixel pairs:

```
p1 ↔ p2, p1' ↔ p2', ...
```

### **3. Essential Matrix Estimation**

Use the matched 2D points and the camera intrinsic matrix to compute the **Essential Matrix**, which encodes the 3D geometry between the two views.

### **4. Recover Relative Pose**

Decompose the Essential Matrix into:

* A **rotation matrix R (3×3)**
* A **translation vector T (3×1)** (up to scale)

Convert rotation to Euler angles (Roll, Pitch, Yaw).
Return all six parameters as the relative motion between the images.

---

## 🛠 Quick Start

### **Prerequisites**

```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### **Run Evaluation Pipelines**

**Phone data evaluation:**
```bash
python -m src.run_phone_data [--step STEP] [--no-plot] [--no-video] [--video-fps FPS]

# Optional arguments:
#   --step STEP  






      Frame step interval (default: 5)
#   --no-plot          Skip plot generation
#   --no-video         Skip video generation
#   --video-fps FPS    Video FPS (default: 5)

# Example with defaults:
python -m src.run_phone_data

# Example with custom step:
python -m src.run_phone_data --step 10 --video-fps 8
```

**Salah Qadah data evaluation:**
This project uses data provided by Salah Qadah (sqadah02@campus.haifa.ac.il) for the purpose of evaluating the model.
```bash
python -m src.run_vo_database_salah [--step STEP] [--no-plot] [--no-video] [--video-fps FPS]

# Optional arguments:
#   --step STEP        Frame step interval (default: 5)
#   --no-plot          Skip plot generation
#   --no-video         Skip video generation
#   --video-fps FPS    Video FPS (default: 5)

# Example with defaults:
python -m src.run_vo_database_salah

# Example with custom step:
python -m src.run_vo_database_salah --step 10 --video-fps 8
```




```bash
python -m src.run_simulator_data [--step STEP] [--no-plot] [--no-video] [--video-fps FPS]

# Optional arguments:
#   --step STEP        Frame step interval (default: 15)
#   --no-plot          Skip plot generation
#   --no-video         Skip video generation
#   --video-fps FPS    Video FPS (default: 10)

# Example with defaults:
python -m src.run_simulator_data

# Example skipping visualization:
python -m src.run_simulator_data --no-plot --no-video
```

**Single pair estimation:**
```bash
python -m src.run_single_pair [--img1 PATH] [--img2 PATH] [--calibration PATH]

# Optional arguments:
#   --img1 PATH              Path to first image (default: evaluation-runs/single-pair/images/000000.png)
#   --img2 PATH              Path to second image (default: evaluation-runs/single-pair/images/000015.png)
#   --calibration PATH       Path to calibration .npz file (default: uses CameraCalibration defaults)
#   -c PATH                  Short form of --calibration

# Example with defaults (uses frames 0 and 15 from single-pair directory):
python -m src.run_single_pair

# Example with custom images:
python -m src.run_single_pair --img1 path/to/img1.png --img2 path/to/img2.png

# Example with custom calibration:
python -m src.run_single_pair -c evaluation-runs/phone-data/data/calibration_scaled.npz

# The return from run_single_pair will be:
## R - Rotation matrix   [3*3] 
## T - Translation vector [3*1]
```


### **Docker Usage**

```bash
# Build Docker image
docker build -t pose-estimator:latest .

# Phone data evaluation (Linux/Mac)
docker run --rm \
  -v $(pwd)/evaluation-runs:/app/evaluation-runs \
  pose-estimator:latest \
  python -m src.run_phone_data

# Simulator data evaluation (Linux/Mac)
docker run --rm \
  -v $(pwd)/evaluation-runs:/app/evaluation-runs \
  pose-estimator:latest \
  python -m src.run_simulator_data

# Single pair estimation (Linux/Mac)
docker run --rm \
  -v $(pwd)/evaluation-runs:/app/evaluation-runs \
  pose-estimator:latest \
  python -m src.run_single_pair

# Running single pair with Windows PowerShell
docker run --rm `
  -v ${PWD}/evaluation-runs:/app/evaluation-runs `
  pose-estimator:latest `
  python -m src.run_single_pair
```

### **Output**

Results are saved to `evaluation-runs/{dataset}/results/`:
- `orientation_plot.html` - 3D trajectory visualization
- `pose_comparison.mp4` - Annotated video
- `evaluation_results.csv` - Error metrics

---

## 📷 Camera Calibration (Phone Data)

The phone data evaluation uses pre-calibrated camera intrinsics stored in `evaluation-runs/phone-data/data/calibration_scaled.npz`.

### **Calibration Process**

The calibration was performed using classical chessboard-based calibration:

1. **Chessboard images captured** - 28 images of a 7×7 chessboard (4cm squares) from various angles
2. **Corner detection** - OpenCV's `findChessboardCorners` + `cornerSubPix` for sub-pixel accuracy
3. **Camera calibration** - `cv2.calibrateCamera` computed intrinsic matrix K and distortion coefficients
4. **Resolution scaling** - K matrix scaled by ~2.36× to match phone data images (848×478 pixels)

### **Calibration Files**

- **Source:** `evaluation-runs/phone-data/camera_calibration_code/`
  - `calibrate.py` - Chessboard calibration script
  - `calibration_filtered.npz` - Original K matrix (high resolution)
  - `chess_calibration/*.jpeg` - 28 chessboard images used for calibration

- **Output:** `evaluation-runs/phone-data/data/calibration_scaled.npz`
  - Contains: Camera matrix K scaled to 848×478 resolution
  - Used by: `run_phone_data.py` for pose estimation

**Note:** The `camera_calibration_code` folder is **reference material** documenting the calibration process. It does not participate in the final workflow but shows how the phone's camera intrinsics were derived.

---

## 🙏 Development Credits

**Architecture & Design:** This solution was architected and designed by the repository owner.

**Development Tools:**
- **ChatGPT** was used for advanced research on the mathematical foundations of multi-view geometry and the OpenCV image processing framework, which made the core image processing pipeline possible.
- **Claude Code** was instrumental in refactoring the codebase to be tight, efficient, and modular, following SOLID principles and best practices.

---

## 🧩 **Mathematical Background (Practical Explanation)**

Although the algorithm relies on epipolar geometry, you do **not** need deep theoretical background.
Here is the practical intuition behind what happens:

### **1. Matching points give clues about camera movement**

If the camera moves:

* forward → points move outward from the center
* right → points shift left
* yaw → points rotate
* roll → entire image tilts

Each matched pair `(p1, p2)` reveals a small piece of information about how the camera moved.

---

### **2. All point pairs must satisfy one geometric constraint**

For two ideal pinhole-camera views of a static scene, corresponding points satisfy:

[
p_2^T , E , p_1 = 0
]

Where:

* (p_1, p_2) are normalized 2D pixel coordinates
* (E) is the **Essential Matrix**
* The equation expresses the fact that the 3D point, the first camera center, and the second camera center lie on a single plane

You don’t need to compute this manually — OpenCV does it for you.

---

### **3. Solving for E extracts camera motion**

OpenCV estimates the matrix (E) from many matched point pairs using RANSAC:

```cpp
E = cv::findEssentialMat(pts1, pts2, K)
```

Since (E) depends only on:

* the direction the camera turned, and
* the direction it translated

we can extract those parameters from it.

---

### **4. RecoverPose decomposes E into R and T**

```cpp
cv::recoverPose(E, pts1, pts2, K, R, T);
```

This gives:

* **R**: rotation matrix (exact)
* **T**: translation vector (up to scale)

Together they describe the full **6 DoF** relative pose:

```
Tx, Ty, Tz, Roll, Pitch, Yaw
```

This is the same approach used in modern SLAM and Visual Odometry systems.

---
