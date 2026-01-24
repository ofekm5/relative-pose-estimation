# Experiment Report – Pose Estimation Validation

## Summary

We continued working on the 6-DoF pose estimation project and evaluated the algorithm using both **simulation** and **real-world QR-based experiments**.

---

## Simulation Results

Using a simulator with known ground truth, the algorithm successfully minimizes the error between the estimated pose and the ground truth, including during camera turns. This confirms that the geometric pipeline functions correctly.
Failures occur mainly under **large perspective changes**, which is consistent with known limitations of ORB-based feature matching.

---

## QR-Based Experiments

### Setup

* Camera intrinsics were calibrated using a chessboard.
* A video of a QR code was recorded and frames were extracted.
* The pose estimation algorithm was applied to consecutive frames.
* QR-based pose was used as a reference.

### Observations

* The algorithm shows inconsistent behavior in this setup.
* A **constant bias in the pitch component** is observed, while roll is also expected to remain near zero.
* Given the setup (camera facing the QR code), both pitch and roll should be approximately zero.
* **Yaw estimates are stable** and close to −180°, which is equivalent to 0°.
* As the camera approaches the QR code, the **Z translation changes as expected**.

---

## Open Questions

* Is the observed pitch bias caused by **calibration inaccuracies**?
* How reliable is **QR-based pose estimation** as ground truth?
* Could there be a **coordinate frame mismatch** between the QR reference and the recovered pose?

---

## Conclusion

The algorithm performs well in simulation and partially in real-world experiments. While yaw and translation behave as expected, systematic pitch (and possibly roll) deviations suggest limitations in calibration accuracy or QR-based ground truth reliability. Further analysis is required to isolate the source of these errors.

---