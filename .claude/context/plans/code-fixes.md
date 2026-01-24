# Remaining Code Fixes

## Critical

### 1. Translation direction error not implemented
**File:** `src/core/pose_evaluator.py:105`
**Fix:** Either implement using GT delta positions or remove the placeholder and field entirely.

### 2. Inefficient angle wrapping
**File:** `src/core/pose_evaluator.py:172-186`
**Fix:** Replace while loops with modulo: `((error_deg + 180) % 360) - 180`

## Medium

### 3. Indentation inconsistency
**File:** `src/core/camera_calibration.py:17-64`
**Fix:** Normalize to 4-space indentation in `__init__` method.

### 4. Ambiguous direction flip comment
**File:** `src/core/visualizer.py:287`
**Fix:** Remove "if needed" comment, document the actual convention.

### 5. Delete redundant geometry_zyx.py
**File:** `src/utils/geometry_zyx.py`
**Fix:** Delete file - functions now in geometry.py.

## Minor

### 6. Missing docstring character
**File:** `src/utils/image_loader.py:1`
**Fix:** Docstring starts at column 1, not 0 (cosmetic).

### 7. No error handling for missing frame
**File:** `src/core/ground_truth_loader.py:56`
**Fix:** Add try/except or check before `iloc[0]` to give clearer error message.
