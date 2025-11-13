## **Camera Intrinsics**

Camera intrinsics describe **how the camera maps 3D rays to 2D pixels**.

They are stored in a 3×3 matrix called **K**.

### The standard form of K:

[
K =
\begin{bmatrix}
f_x & 0   & c_x \
0   & f_y & c_y \
0   & 0   & 1
\end{bmatrix}
]

Where:

### **fx, fy**

* the focal length of the camera in pixels
* typically same value unless the pixels are rectangular

### **cx, cy**

* the principal point (usually at the image center)

### Example:

For a 640×480 image:

```cpp
cv::Mat K = (cv::Mat_<double>(3, 3) <<
    500, 0,   320,
    0,   500, 240,
    0,   0,   1);
```

Interpretation:

* `fx = 500 px`
* `fy = 500 px`
* `cx = 320` → image width/2
* `cy = 240` → image height/2

### Why do we need intrinsics?

They allow OpenCV to convert pixel coordinates into normalized camera rays so that the Essential Matrix can be translated into actual camera motion.

Without K:

* the math is ambiguous
* relative pose cannot be computed correctly

### Do you need to calibrate your camera?

**Not necessarily for this project.**

You can:

* use an approximate K
* assume the principal point is the image center
* assume focal length between 450–700 px for cheap cameras

Your project will still work because:

* translation is returned **up to scale**
* rotation is very stable even with approximate K

---
