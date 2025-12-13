# Test Scripts

This folder contains test and evaluation scripts for the pose estimation system.

## Available Scripts

### 1. `run_evaluation.py`
Run complete pose estimation evaluation with visualization.

**Usage:**
```bash
python tests/run_evaluation.py [data_dir] [step]

# Examples:
python tests/run_evaluation.py silmulator_data/simple_movement 15
python tests/run_evaluation.py silmulator_data/simple_movement 30

# Via Makefile:
make run-eval
make run-eval STEP=30
```

**Parameters:**
- `data_dir`: Path to data directory (default: `silmulator_data/simple_movement`)
- `step`: Frame interval for evaluation (default: `15`)

---

### 2. `run_video.py`
Generate video with ground truth vs estimated trajectory overlay.

**Usage:**
```bash
python tests/run_video.py [data_dir] [output_path] [step] [fps]

# Examples:
python tests/run_video.py silmulator_data/simple_movement results/output.mp4 15 10
python tests/run_video.py silmulator_data/simple_movement results/demo.mp4 30 5

# Via Makefile:
make run-video
make run-video STEP=30
```

**Parameters:**
- `data_dir`: Path to data directory (default: `silmulator_data/simple_movement`)
- `output_path`: Output video path (default: `results/output.mp4`)
- `step`: Frame interval (default: `15`)
- `fps`: Video framerate (default: `10`)

---

### 3. `run_single_pair.py`
Run pose estimation on a single frame pair.

**Usage:**
```bash
python tests/run_single_pair.py [data_dir] [frame1] [frame2]

# Examples:
python tests/run_single_pair.py silmulator_data/simple_movement 0 15
python tests/run_single_pair.py silmulator_data/simple_movement 100 150

# Via Makefile:
make run-single
```

**Parameters:**
- `data_dir`: Path to data directory (default: `silmulator_data/simple_movement`)
- `frame1`: First frame index (default: `0`)
- `frame2`: Second frame index (default: `15`)

**Output:**
```
Estimated Pose:
  Yaw:   12.34°
  Pitch: -5.67°
  Roll:  0.89°
```

---

## Running Tests via Makefile

All test scripts can be run using convenient Makefile targets:

```bash
# Install dependencies first
make install

# Run evaluation
make run-eval

# Generate plots (same as run-eval)
make run-plot

# Generate video
make run-video

# Run single frame pair
make run-single

# Customize parameters
make run-eval DATA_DIR=silmulator_data/simple_movement STEP=30
make run-video STEP=20 RESULTS_DIR=my_results
```

## Running Tests in Docker

```bash
# Build Docker image
make docker-build

# Run evaluation inside Docker
make docker-eval

# Run custom test in Docker
docker run --rm \
  -v $(pwd)/silmulator_data:/app/silmulator_data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/tests:/app/tests \
  pose-estimator:latest \
  python tests/run_single_pair.py silmulator_data/simple_movement 0 30
```

## Directory Structure

```
tests/
├── README.md              # This file
├── __init__.py           # Python package marker
├── run_evaluation.py     # Full evaluation with visualization
├── run_video.py          # Video generation
└── run_single_pair.py    # Single frame pair test
```

## Adding New Tests

To add a new test script:

1. Create a new `.py` file in this directory
2. Add a shebang: `#!/usr/bin/env python3`
3. Add command-line argument parsing
4. Import from `src/` modules
5. Add a new Makefile target (optional)

Example template:

```python
#!/usr/bin/env python3
"""
Description of your test.
"""

import sys
from src.pose_matcher import PoseMatcher

def main():
    # Your test code here
    pass

if __name__ == "__main__":
    main()
```
