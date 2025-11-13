## **What is MatchResult? What do pts1 and pts2 mean?**

This structure stores **the corresponding 2D points** in both images.

### Declared as:

```cpp
struct MatchResult {
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
};
```

### Meaning:

* `pts1[i]` = 2D pixel location of a feature in **image 1**
* `pts2[i]` = the **matching** 2D pixel location of the **same physical point** in **image 2**

They **must** be aligned by index:

```
pts1[0] ↔ pts2[0]
pts1[1] ↔ pts2[1]
pts1[2] ↔ pts2[2]
...
```

Each pair represents a real-world feature that appears in both images.

---

# ⭐ **Example of MatchResult**

Say image 1 has a corner at pixel (100, 150), and the same corner in image 2 moved to (120, 148) due to camera movement.

Then:

```cpp
pts1 = { cv::Point2f(100, 150) }
pts2 = { cv::Point2f(120, 148) }
```

If OpenCV found 4 matching feature pairs:

```cpp
pts1 = {
    (100, 150),
    (230,  80),
    ( 50, 250),
    (300, 220)
}

pts2 = {
    (120, 148),
    (240,  78),
    ( 55, 245),
    (318, 215)
}
```

Notice how each pair represents the **same real 3D point**, seen from two slightly different camera positions.

### Why do we need these?

`cv::findEssentialMat()` and `cv::recoverPose()` use these point pairs to compute:

* how the camera moved (Translation)
* how the camera rotated (Rotation)

These point pairs are the **only input** needed for geometric pose recovery.

---