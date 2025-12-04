.PHONY: help build build-cpu build-tf build-tf-cpu push push-cpu push-tf push-tf-cpu push-fastattn shell test clean az-login apptainer-build apptainer-build-torch-gpu apptainer-build-torch-cpu apptainer-build-tensorflow-gpu apptainer-build-tensorflow-cpu apptainer-build-fastattn apptainer-shell apptainer-run apptainer-save-models apptainer-save-all-models apptainer-save-patch-models apptainer-save-slide-models apptainer-tessellate apptainer-extract-features apptainer-clean apptainer-examples

# Default Docker image name
IMAGE_NAME ?= mskocracontainerregister-cfbfchg8dgfbedan.azurecr.io/mussel
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

build-fastattn: ## Build Docker image with flash-attention support
	$(MAKE) build BACKEND=fastattn IMAGE_TAG=fastattn-dev

push: az-login ## Push Docker image to registry
	@echo "Pushing Docker image: $(FULL_IMAGE)"
	docker push $(FULL_IMAGE)

push-cpu: az-login ## Push Docker image with CPU support to registry
	$(MAKE) push IMAGE_TAG=cpu

push-tf: az-login ## Push Docker image with TensorFlow GPU support to registry
	$(MAKE) push IMAGE_TAG=tf-gpu

push-tf-cpu: az-login ## Push Docker image with TensorFlow CPU support to registry
	$(MAKE) push IMAGE_TAG=tf-cpu

push-fastattn: az-login ## Push Docker image with flash-attention support to registry
	$(MAKE) push IMAGE_TAG=fastattn-dev

az-login:
	az acr login -n mskOcraContainerRegister

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

# Apptainer/Singularity commands
APPTAINER_IMAGE ?= mussel_$(IMAGE_TAG).sif
DOCKER_URI ?= docker://$(FULL_IMAGE)
APPTAINER_CACHE_DIR ?= $(HOME)/.apptainer/cache
MODEL_DIR ?= ./model_cache

apptainer-build: ## Build Apptainer/Singularity image from Docker image
	@echo "Building Apptainer image: $(APPTAINER_IMAGE) from $(DOCKER_URI)"
	apptainer build $(APPTAINER_IMAGE) $(DOCKER_URI)

apptainer-build-torch-gpu: ## Build Apptainer image with PyTorch GPU backend (via Docker)
	@echo "Building Docker image with torch-gpu backend"
	docker build --build-arg BACKEND=torch-gpu -t $(IMAGE_NAME):torch-gpu .
	@echo "Converting Docker image to Apptainer SIF"
	apptainer build mussel_torch-gpu.sif docker-daemon://$(IMAGE_NAME):torch-gpu

apptainer-build-torch-cpu: ## Build Apptainer image with PyTorch CPU backend (via Docker)
	@echo "Building Docker image with torch-cpu backend"
	docker build --build-arg BACKEND=torch-cpu -t $(IMAGE_NAME):torch-cpu .
	@echo "Converting Docker image to Apptainer SIF"
	apptainer build mussel_torch-cpu.sif docker-daemon://$(IMAGE_NAME):torch-cpu

apptainer-build-tensorflow-gpu: ## Build Apptainer image with TensorFlow GPU backend (via Docker)
	@echo "Building Docker image with tensorflow-gpu backend"
	docker build --build-arg BACKEND=tensorflow-gpu -t $(IMAGE_NAME):tensorflow-gpu .
	@echo "Converting Docker image to Apptainer SIF"
	apptainer build mussel_tensorflow-gpu.sif docker-daemon://$(IMAGE_NAME):tensorflow-gpu

apptainer-build-tensorflow-cpu: ## Build Apptainer image with TensorFlow CPU backend (via Docker)
	@echo "Building Docker image with tensorflow-cpu backend"
	docker build --build-arg BACKEND=tensorflow-cpu -t $(IMAGE_NAME):tensorflow-cpu .
	@echo "Converting Docker image to Apptainer SIF"
	apptainer build mussel_tensorflow-cpu.sif docker-daemon://$(IMAGE_NAME):tensorflow-cpu

apptainer-build-fastattn: ## Build Apptainer image with flash-attention backend (via Docker)
	@echo "Building Docker image with fastattn backend"
	docker build --build-arg BACKEND=fastattn -t $(IMAGE_NAME):fastattn .
	@echo "Converting Docker image to Apptainer SIF"
	apptainer build mussel_fastattn.sif docker-daemon://$(IMAGE_NAME):fastattn

apptainer-shell: ## Start interactive shell in Apptainer container
	@echo "Starting Apptainer shell with GPU support"
	apptainer shell --nv --bind $(PWD):/workspace $(APPTAINER_IMAGE)

apptainer-run: ## Run command in Apptainer container (usage: make apptainer-run CMD="your command")
	@if [ -z "$(CMD)" ]; then \
		echo "Error: CMD variable not set. Usage: make apptainer-run CMD=\"your command\""; \
		exit 1; \
	fi
	apptainer exec --nv --bind $(PWD):/workspace --pwd /workspace $(APPTAINER_IMAGE) $(CMD)

apptainer-save-models: ## Save specific models (usage: make apptainer-save-models MODELS="UNI2,VIRCHOW2")
	@if [ -z "$(MODELS)" ]; then \
		echo "Error: MODELS variable not set. Usage: make apptainer-save-models MODELS=\"UNI2,VIRCHOW2\""; \
		exit 1; \
	fi
	@echo "Saving models: $(MODELS) to $(MODEL_DIR)"
	mkdir -p $(MODEL_DIR)
	mkdir -p .cache/huggingface
	apptainer exec --nv \
		--bind $(PWD):/workspace \
		--pwd /workspace \
		--env HF_HOME=/workspace/.cache/huggingface \
		--env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
		$(APPTAINER_IMAGE) \
		save_model \
		model_types=[$(MODELS)] \
		model_dir=$(MODEL_DIR)

apptainer-save-all-models: ## Save all supported models to model_cache directory
	@echo "Saving all models to $(MODEL_DIR)"
	mkdir -p $(MODEL_DIR)
	mkdir -p .cache/huggingface
	apptainer exec --nv \
		--bind $(PWD):/workspace \
		--pwd /workspace \
		--env HF_HOME=/workspace/.cache/huggingface \
		--env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
		$(APPTAINER_IMAGE) \
		save_model \
		model_types=[CLIP,CTRANSPATH,GIGAPATH,GIGAPATH_SLIDE,GOOGLEPATH,OPTIMUS,TITAN_SLIDE,UNI,UNI2,VIRCHOW,VIRCHOW2] \
		model_dir=$(MODEL_DIR)

apptainer-save-patch-models: ## Save all patch-level encoder models
	@echo "Saving patch encoder models to $(MODEL_DIR)"
	mkdir -p $(MODEL_DIR)
	mkdir -p .cache/huggingface
	apptainer exec --nv \
		--bind $(PWD):/workspace \
		--pwd /workspace \
		--env HF_HOME=/workspace/.cache/huggingface \
		--env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
		$(APPTAINER_IMAGE) \
		save_model \
		model_types=[CTRANSPATH,GIGAPATH,OPTIMUS,UNI,UNI2,VIRCHOW,VIRCHOW2,CLIP] \
		model_dir=$(MODEL_DIR)

apptainer-save-slide-models: ## Save all slide-level encoder models
	@echo "Saving slide encoder models to $(MODEL_DIR)"
	mkdir -p $(MODEL_DIR)
	mkdir -p .cache/huggingface
	apptainer exec --nv \
		--bind $(PWD):/workspace \
		--pwd /workspace \
		--env HF_HOME=/workspace/.cache/huggingface \
		--env TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
		$(APPTAINER_IMAGE) \
		save_model \
		model_types=[GIGAPATH_SLIDE,TITAN_SLIDE] \
		model_dir=$(MODEL_DIR)

apptainer-tessellate: ## Run tessellate with Apptainer (usage: make apptainer-tessellate SLIDE=slide.svs)
	@if [ -z "$(SLIDE)" ]; then \
		echo "Error: SLIDE variable not set. Usage: make apptainer-tessellate SLIDE=slide.svs"; \
		exit 1; \
	fi
	apptainer exec --nv \
		--bind $(PWD):/workspace \
		--pwd /workspace \
		$(APPTAINER_IMAGE) \
		uv run tessellate slide_path=$(SLIDE) output_h5_path=$(basename $(SLIDE)).h5

apptainer-extract-features: ## Run extract features with Apptainer (usage: make apptainer-extract-features SLIDE=slide.svs MODEL=UNI2)
	@if [ -z "$(SLIDE)" ] || [ -z "$(MODEL)" ]; then \
		echo "Error: SLIDE and MODEL variables required"; \
		echo "Usage: make apptainer-extract-features SLIDE=slide.svs MODEL=UNI2"; \
		exit 1; \
	fi
	mkdir -p $(MODEL_DIR)
	apptainer exec --nv \
		--bind $(PWD):/workspace \
		--bind $(MODEL_DIR):/models \
		--pwd /workspace \
		--env MODEL_DIR=/models \
		$(APPTAINER_IMAGE) \
		uv run tessellate_extract_features \
		slide_paths=[$(SLIDE)] \
		model_type=$(MODEL) \
		output_dir=output \
		model_dir=/models

apptainer-clean: ## Remove Apptainer images and cache
	rm -f *.sif
	rm -rf $(APPTAINER_CACHE_DIR)

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

apptainer-examples: ## Show example Apptainer commands
	@echo "Apptainer Example commands:"
	@echo ""
	@echo "  # Build Apptainer image from Docker registry"
	@echo "  make apptainer-build"
	@echo ""
	@echo "  # Build Apptainer image directly from Dockerfile (backend-specific)"
	@echo "  make apptainer-build-torch-gpu       # PyTorch with GPU support"
	@echo "  make apptainer-build-torch-cpu       # PyTorch CPU only"
	@echo "  make apptainer-build-tensorflow-gpu  # TensorFlow with GPU"
	@echo "  make apptainer-build-tensorflow-cpu  # TensorFlow CPU only"
	@echo "  make apptainer-build-fastattn        # PyTorch with flash-attention"
	@echo ""
	@echo "  # Save models to model_cache directory"
	@echo "  make apptainer-save-all-models                          # Save all 11 models"
	@echo "  make apptainer-save-patch-models                        # Save patch encoders only"
	@echo "  make apptainer-save-slide-models                        # Save slide encoders only"
	@echo "  make apptainer-save-models MODELS=\"UNI2,VIRCHOW2\"       # Save specific models"
	@echo "  make apptainer-save-models MODELS=\"GIGAPATH_SLIDE\" MODEL_DIR=/path/to/models"
	@echo ""
	@echo "  # Start interactive shell"
	@echo "  make apptainer-shell"
	@echo ""
	@echo "  # Run tessellate"
	@echo "  make apptainer-tessellate SLIDE=slide.svs"
	@echo ""
	@echo "  # Extract features"
	@echo "  make apptainer-extract-features SLIDE=slide.svs MODEL=UNI2"
	@echo ""
	@echo "  # Run custom command"
	@echo "  make apptainer-run CMD=\"save_model --help\""
	@echo ""
	@echo "  # Clean up images"
	@echo "  make apptainer-clean"
