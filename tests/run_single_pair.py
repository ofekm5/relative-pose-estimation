#!/usr/bin/env python3
"""
Run pose estimation on a single frame pair.

Usage:
    python tests/run_single_pair.py [data_dir] [frame1] [frame2]

Example:
    python tests/run_single_pair.py silmulator_data/simple_movement 0 15
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.pose_matcher import PoseMatcher
from src.image_loader import load_image_pair


def main():
    # Default values
    data_dir = "silmulator_data/simple_movement"
    frame1_idx = 0
    frame2_idx = 15

    # Parse command line arguments
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        frame1_idx = int(sys.argv[2])
    if len(sys.argv) > 3:
        frame2_idx = int(sys.argv[3])

    # Construct paths
    gt_path = os.path.join(data_dir, "camera_poses.txt")
    img1_path = os.path.join(data_dir, "images", f"{frame1_idx:06d}.png")
    img2_path = os.path.join(data_dir, "images", f"{frame2_idx:06d}.png")

    print(f"Data directory: {data_dir}")
    print(f"Frame 1: {img1_path}")
    print(f"Frame 2: {img2_path}")
    print(f"Ground truth: {gt_path}")
    print("-" * 50)

    # Initialize matcher
    matcher = PoseMatcher(data_dir, gt_path)

    # Load images
    img1, img2 = load_image_pair(img1_path, img2_path, to_gray=True)

    # Estimate pose
    yaw, pitch, roll = matcher.match(img1, img2, prev_frame_index=frame1_idx)

    print("-" * 50)
    print("Estimated Pose:")
    print(f"  Yaw:   {yaw:.2f}°")
    print(f"  Pitch: {pitch:.2f}°")
    print(f"  Roll:  {roll:.2f}°")
    print("-" * 50)
    print("✓ Pose estimation complete!")


if __name__ == "__main__":
    main()
