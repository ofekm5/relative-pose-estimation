# relative-pose-estimation

### **Classical Multi-View Geometry — No Training Required**

This repository implements a complete system for estimating **6 Degrees of Freedom (6-DoF)** camera motion between two JPEG images taken consecutively by a moving robot (e.g., drone, car, rover).

The method uses **ORB features**, **feature matching**, **Essential Matrix estimation**, and **RecoverPose** to compute:

* **Translation:** Tx, Ty, Tz
* **Rotation:** Roll, Pitch, Yaw

This pipeline requires **no dataset** and **no learning**, and works entirely with classical geometric computer-vision methods implemented in OpenCV.

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

## 📦 Project Structure

```
relative-pose-estimation-opencv/
│
├── src/
│   ├── main.cpp              # Entry point
│   ├── orb_features.cpp      # ORB keypoints + descriptors
│   ├── match_features.cpp    # Feature matching
│   ├── essential_pose.cpp    # Essential matrix + RecoverPose
│   ├── rotation_utils.cpp    # Convert R → Euler angles
│   ├── camera.cpp            # Camera intrinsics handling
│   └── utils/                # Extra helper functions
│
├── include/                  # Header files
├── Dockerfile                # ARM-compatible OpenCV build
└── README.md
```

---

## 🛠 Build & Run using Docker

### **1. Build Docker image**

```
docker build -t pose-estimator .
```

This image contains:

* OpenCV (compiled with ARM/NEON optimizations)
* The C++ build of this project
* A minimal runtime environment for Raspberry Pi or x86

---

### **2. Run**

```
docker run --rm \
  -v $(pwd)/images:/data \
  pose-estimator \
  /data/before.jpg /data/after.jpg
```

The output format is:

```
Tx: ...
Ty: ...
Tz: ...
Roll: ...
Pitch: ...
Yaw: ...
```

---
