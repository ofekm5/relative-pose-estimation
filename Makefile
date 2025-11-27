# Makefile wrapper for CMake-based build system
# Simple orchestration for local builds and Docker

IMAGE_NAME := pose-estimator
BUILD_DIR := build

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make build                  - Build project locally using CMake"
	@echo "  make clean                  - Clean build directory"
	@echo "  make docker-build           - Build Docker image"
	@echo "  make docker-run             - Run Docker container with example images"
	@echo "  make docker-run-interactive - Open bash shell inside Docker container"
	@echo "  make run IMG1=... IMG2=...  - Run locally built executable"

.PHONY: build
build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && cmake --build .
	@echo "Build complete: $(BUILD_DIR)/pose_estimator"

.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned"

.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE_NAME):latest .

.PHONY: docker-run
docker-run:
	docker run --rm -v "$$(pwd)/images:/data" $(IMAGE_NAME):latest /data/before.jpg /data/after.jpg

.PHONY: docker-run-interactive
docker-run-interactive:
	docker run --rm -it -v "$$(pwd)/images:/data" --entrypoint /bin/bash $(IMAGE_NAME):latest

.PHONY: run
run:
	@$(BUILD_DIR)/pose_estimator $(IMG1) $(IMG2)

.DEFAULT_GOAL := help
