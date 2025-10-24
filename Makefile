.PHONY: help build build-cpu build-tf build-tf-cpu shell test clean

# Default Docker image name
IMAGE_NAME ?= mussel
IMAGE_TAG ?= latest
FULL_IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)

# Backend options: torch-gpu, torch-cpu, tensorflow-gpu, tensorflow-cpu
BACKEND ?= torch-gpu

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build Docker image with PyTorch GPU support (default)
	@echo "Building Docker image: $(FULL_IMAGE) with backend: $(BACKEND)"
	docker build --build-arg BACKEND=$(BACKEND) -t $(FULL_IMAGE) .

build-cpu: ## Build Docker image with PyTorch CPU support
	$(MAKE) build BACKEND=torch-cpu IMAGE_TAG=cpu

build-tf: ## Build Docker image with TensorFlow GPU support
	$(MAKE) build BACKEND=tensorflow-gpu IMAGE_TAG=tf-gpu

build-tf-cpu: ## Build Docker image with TensorFlow CPU support
	$(MAKE) build BACKEND=tensorflow-cpu IMAGE_TAG=tf-cpu

shell: ## Start an interactive shell in the container
	docker run --rm -it \
		--user $(shell id -u):$(shell id -g) \
		--gpus all \
		-v $(PWD):/data \
		-w /data \
		$(FULL_IMAGE) \
		/bin/bash

shell-cpu: ## Start an interactive shell (CPU only)
	docker run --rm -it \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD):/data \
		-w /data \
		$(IMAGE_NAME):cpu \
		/bin/bash

test: ## Run tests in Docker container
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-v $(PWD):/code/mussel \
		-w /code/mussel \
		$(FULL_IMAGE) \
		uv run pytest tests

clean: ## Remove Docker images
	docker rmi $(FULL_IMAGE) || true
	docker rmi $(IMAGE_NAME):cpu || true
	docker rmi $(IMAGE_NAME):tf-gpu || true
	docker rmi $(IMAGE_NAME):tf-cpu || true

# Example commands
examples: ## Show example Docker commands
	@echo "Example commands:"
	@echo ""
	@echo "  # Build and run tessellate"
	@echo "  make build"
	@echo "  docker run --rm --user \$$(id -u):\$$(id -g) --gpus all -v \$$(pwd):/data -w /data $(FULL_IMAGE) \\"
	@echo "    uv run tessellate slide_path=slide.svs output_h5_path=tiles.h5"
	@echo ""
	@echo "  # Extract features"
	@echo "  docker run --rm --user \$$(id -u):\$$(id -g) --gpus all -v \$$(pwd):/data -w /data $(FULL_IMAGE) \\"
	@echo "    uv run extract_features slide_path=slide.svs patch_h5_path=tiles.h5 \\"
	@echo "    model_type=CLIP output_h5_path=features.h5"
	@echo ""
	@echo "  # Or use the mussel-docker wrapper:"
	@echo "  ./mussel-docker tessellate slide_path=slide.svs output_h5_path=tiles.h5"
