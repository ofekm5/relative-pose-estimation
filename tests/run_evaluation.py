#!/usr/bin/env python3
"""
Run pose estimation evaluation with visualization.

Usage:
    python tests/run_evaluation.py [data_dir] [step]

Example:
    python tests/run_evaluation.py silmulator_data/simple_movement 15
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
    step = 15

    # Parse command line arguments
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        step = int(sys.argv[2])

    print(f"Running evaluation on: {data_dir}")
    print(f"Frame step: {step}")
    print("-" * 50)

    # Create plotter and run evaluation
    plotter = PosePlotter(data_dir, step=step)
    plotter.run()

    print("-" * 50)
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    main()
