#!/usr/bin/env python3
"""
Generate video with ground truth overlay.

Usage:
    python tests/run_video.py [data_dir] [output_path] [step] [fps]

Example:
    python tests/run_video.py silmulator_data/simple_movement results/output.mp4 15 10
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.plots_graths import PosePlotter


def main():
    # Default values
    data_dir = "silmulator_data/simple_movement"
    output_path = "results/output.mp4"
    step = 15
    fps = 10

    # Parse command line arguments
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    if len(sys.argv) > 3:
        step = int(sys.argv[3])
    if len(sys.argv) > 4:
        fps = int(sys.argv[4])

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Generating video from: {data_dir}")
    print(f"Output path: {output_path}")
    print(f"Frame step: {step}")
    print(f"FPS: {fps}")
    print("-" * 50)

    # Create plotter and generate video
    plotter = PosePlotter(data_dir, step=step)
    plotter.make_video(output_path, fps=fps)

    print("-" * 50)
    print(f"✓ Video saved to: {output_path}")


if __name__ == "__main__":
    main()
