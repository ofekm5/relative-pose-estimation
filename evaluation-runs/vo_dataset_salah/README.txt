================================================================================
                    Visual Odometry Test Dataset
                         Salah - January 2025
================================================================================

Dataset for testing 2-frame Visual Odometry algorithms with ground truth.

--------------------------------------------------------------------------------
CONTENTS
--------------------------------------------------------------------------------

1. frames/
   - 477 JPEG images (1920x1080, 30fps)

2. calibration/
   - calibration.npz       : Camera intrinsic matrix and distortion coefficients
   - calibration.txt       : Human-readable calibration values

3. ground_truth/
   - ground_truth.npz      : Camera trajectory (positions + headings per frame)
   - ground_truth.txt      : Human-readable format (frame, x, y, z, heading)

4. markers/
   - marker_positions.txt  : ArUco marker world coordinates
   - marker_layout.png     : Visual diagram of marker placement (if available)

--------------------------------------------------------------------------------
CAMERA CALIBRATION
--------------------------------------------------------------------------------

Camera: iPhone 16 Plus (1920x1080)

Intrinsic Matrix K:
    [1888.58,    0.00,  930.37]
    [   0.00, 1900.82,  561.70]
    [   0.00,    0.00,    1.00]

For VO command line:
    -k 1888.58,1900.82,930.37,561.70

--------------------------------------------------------------------------------
GROUND TRUTH GENERATION
--------------------------------------------------------------------------------

Ground truth was generated using ArUco marker detection:
- Dictionary: DICT_6X6_250
- Marker size: 15cm x 15cm
- 5 markers placed at known world positions
- Camera pose estimated via solvePnP when markers visible
- Trajectory smoothed to remove detection noise

--------------------------------------------------------------------------------
COORDINATE SYSTEM
--------------------------------------------------------------------------------

World coordinates (meters):
- X: horizontal (right from marker 0)
- Y: horizontal (forward from marker 0)
- Z: vertical (height from floor)

Marker 0 defines the origin reference point.

--------------------------------------------------------------------------------
USAGE EXAMPLE
--------------------------------------------------------------------------------

# Run VO on consecutive frames
./vo_submission frames/frame_0001.jpg frames/frame_0002.jpg -k 1888.58,1900.82,930.37,561.70

# Compare with ground truth
# Ground truth provides (x, y, z, heading) for each frame index

--------------------------------------------------------------------------------
CONTACT
--------------------------------------------------------------------------------

Created by: Salah
Course: Project in Advanced Robotics
