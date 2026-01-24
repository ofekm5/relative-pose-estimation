# Experiment Update – Initial Results and Analysis

## Summary

This report presents our **initial experimental results** for the relative pose estimation pipeline.

---

## Method

At each step, we estimate the camera’s **relative orientation** using:

* the previous image,
* the previous orientation, and
* the current image.

We assume:

* unit step length between frames,
* constant height (no vertical motion).

The graph shows the accumulated orientation over successive steps.

---

## Initial Findings

A significant observation at this stage is that:

* **Sharp changes in yaw (in-place turns)** lead to **large deviations in the estimated yaw angle**.
* This deviation appears consistently during sudden rotations.

Our current goal is to analyze this behavior in more detail.

---

## Ongoing Investigation

We are focusing on the following questions:

* Is the yaw deviation caused by the **core algorithm** or by a specific stage in the pipeline?
* Does a similar deviation occur for **pitch or roll** changes?
* Do **height variations** introduce additional bias or instability in the estimated angles?

---

## Runtime Considerations

The current experiments are executed on a **laptop**, so measured runtimes may not reflect performance on **Raspberry Pi 5**, which is significantly more constrained.

For this reason, we believe it is beneficial to:

* begin adapting and testing the code on the target hardware,
* identify early challenges related to CPU/GPU load, memory constraints, and library support.

---

