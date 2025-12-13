# Makefile wrapper for Python-based pose estimation system
# Simple orchestration for local development and Docker

IMAGE_NAME := pose-estimator
PYTHON := python3
VENV := venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
DATA_DIR := silmulator_data/simple_movement
RESULTS_DIR := results
STEP := 15

# Automatically use venv if it exists or if we're in a virtual environment
ifdef VIRTUAL_ENV
    PYTHON_CMD := python
    PIP_CMD := pip
else
    PYTHON_CMD := $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),$(PYTHON))
    PIP_CMD := $(if $(wildcard $(VENV_PIP)),$(VENV_PIP),$(PYTHON) -m pip)
endif

.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@echo "Setup:"
	@echo "  make venv                   - Create Python virtual environment"
	@echo "  make install                - Install Python dependencies (uses venv if available)"
	@echo ""
	@echo "Local Development:"
	@echo "  make run-eval               - Run pose estimation evaluation"
	@echo "  make run-plot               - Generate 3D trajectory plots"
	@echo "  make run-video              - Generate video with ground truth overlay"
	@echo "  make clean                  - Remove Python cache files"
	@echo "  make clean-all              - Remove cache and virtual environment"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build           - Build Docker image"
	@echo "  make docker-build ENTRY_FILE=src/plots_graths.py - Build with entry file"
	@echo "  make docker-run             - Run Docker container (interactive Python)"
	@echo "  make docker-shell           - Open bash shell inside Docker container"
	@echo "  make docker-eval            - Run evaluation inside Docker"
	@echo ""
	@echo "Parameters:"
	@echo "  STEP=N                      - Frame interval for evaluation (default: 15)"
	@echo "  DATA_DIR=path               - Path to data directory (default: silmulator_data/simple_movement)"
	@echo "  ENTRY_FILE=path             - Python file to run in Docker build"

.PHONY: venv
venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV); \
		echo "✓ Virtual environment created at $(VENV)"; \
		echo ""; \
		echo "To activate it manually, run:"; \
		echo "  source $(VENV)/bin/activate"; \
	else \
		echo "✓ Virtual environment already exists at $(VENV)"; \
	fi

.PHONY: install
install: venv
	@echo "Installing Python dependencies..."
	@$(PIP_CMD) install -r requirements.txt
	@echo "✓ Dependencies installed"
	@echo ""
	@if [ -z "$$VIRTUAL_ENV" ] && [ -d "$(VENV)" ]; then \
		echo "Note: To activate the virtual environment, run:"; \
		echo "  source $(VENV)/bin/activate"; \
	fi

.PHONY: run-eval
run-eval:
	@$(PYTHON_CMD) tests/run_evaluation.py $(DATA_DIR) $(STEP)

.PHONY: run-plot
run-plot:
	@$(PYTHON_CMD) tests/run_evaluation.py $(DATA_DIR) $(STEP)

.PHONY: run-video
run-video:
	@mkdir -p $(RESULTS_DIR)
	@$(PYTHON_CMD) tests/run_video.py $(DATA_DIR) $(RESULTS_DIR)/output.mp4 $(STEP) 10

.PHONY: run-single
run-single:
	@$(PYTHON_CMD) tests/run_single_pair.py $(DATA_DIR) 0 15

.PHONY: clean
clean:
	@echo "Cleaning Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cache cleaned"

.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "✓ Virtual environment removed"

.PHONY: docker-build
docker-build:
ifdef ENTRY_FILE
	@echo "Building Docker image with entry file: $(ENTRY_FILE)"
	@docker build --build-arg ENTRY_FILE=$(ENTRY_FILE) -t $(IMAGE_NAME):latest .
else
	@echo "Building Docker image (interactive mode)"
	@docker build -t $(IMAGE_NAME):latest .
endif
	@echo "✓ Docker image built: $(IMAGE_NAME):latest"

.PHONY: docker-run
docker-run:
	@echo "Running Docker container (interactive Python shell)..."
	docker run --rm -it \
		-v "$$(pwd)/$(DATA_DIR):/app/$(DATA_DIR)" \
		-v "$$(pwd)/$(RESULTS_DIR):/app/$(RESULTS_DIR)" \
		$(IMAGE_NAME):latest

.PHONY: docker-shell
docker-shell:
	@echo "Opening bash shell in Docker container..."
	docker run --rm -it \
		-v "$$(pwd)/$(DATA_DIR):/app/$(DATA_DIR)" \
		-v "$$(pwd)/$(RESULTS_DIR):/app/$(RESULTS_DIR)" \
		--entrypoint /bin/bash \
		$(IMAGE_NAME):latest

.PHONY: docker-eval
docker-eval:
	@echo "Running evaluation inside Docker container..."
	docker run --rm \
		-v "$$(pwd)/$(DATA_DIR):/app/$(DATA_DIR)" \
		-v "$$(pwd)/$(RESULTS_DIR):/app/$(RESULTS_DIR)" \
		-v "$$(pwd)/tests:/app/tests" \
		$(IMAGE_NAME):latest \
		python tests/run_evaluation.py $(DATA_DIR) $(STEP)
	@echo "✓ Docker evaluation complete"

.DEFAULT_GOAL := help
