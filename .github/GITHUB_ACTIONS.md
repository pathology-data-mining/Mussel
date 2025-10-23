# GitHub Actions Setup

This repository uses GitHub Actions for continuous integration and Docker image deployment to DockerHub.

## Workflows

### CI (Continuous Integration)
- **File**: `.github/workflows/ci.yml`
- **Triggers**: Push to `main` branch, Pull Requests to `main`
- **Purpose**: Run automated tests using pytest
- **Python Versions Tested**: 3.10, 3.11
- **Backend**: Uses `torch-cpu` for faster CI runs

### Docker Build and Push
- **File**: `.github/workflows/docker.yml`
- **Triggers**: 
  - Push to `main` branch
  - Git tags (e.g., `v1.0.0`)
  - Manual workflow dispatch
- **Purpose**: Build and push Docker images to DockerHub
- **Backends Built**:
  - `torch-gpu` - PyTorch with CUDA support
  - `torch-cpu` - PyTorch CPU-only
  - `tensorflow-gpu` - TensorFlow with CUDA support
  - `tensorflow-cpu` - TensorFlow CPU-only

## Required GitHub Secrets

To enable Docker image deployment, you need to set up the following secrets in your GitHub repository:

1. **DOCKERHUB_USERNAME**: Your DockerHub username
2. **DOCKERHUB_TOKEN**: A DockerHub access token (not your password)

### Setting up DockerHub Secrets

1. Go to your GitHub repository settings
2. Navigate to **Settings > Secrets and variables > Actions**
3. Click **New repository secret**
4. Add the following secrets:
   - Name: `DOCKERHUB_USERNAME`, Value: Your DockerHub username
   - Name: `DOCKERHUB_TOKEN`, Value: Your DockerHub access token

### Creating a DockerHub Access Token

1. Log in to [Docker Hub](https://hub.docker.com/)
2. Go to **Account Settings > Security**
3. Click **New Access Token**
4. Give it a descriptive name (e.g., "GitHub Actions")
5. Copy the token and save it as the `DOCKERHUB_TOKEN` secret

## Docker Image Tags

Docker images are tagged using the following scheme:

- `latest-<backend>`: Latest build from the main branch (e.g., `latest-torch-gpu`)
- `main-<backend>`: Latest build from the main branch (same as latest)
- `v1.0.0-<backend>`: Specific version tag (e.g., `v1.0.0-torch-gpu`)
- `1.0-<backend>`: Major.minor version (e.g., `1.0-torch-gpu`)

### Example Docker Pull Commands

```bash
# Pull latest torch-gpu image
docker pull <username>/mussel:latest-torch-gpu

# Pull specific version with torch-cpu
docker pull <username>/mussel:v1.1.1-torch-cpu

# Pull tensorflow-gpu image
docker pull <username>/mussel:latest-tensorflow-gpu
```

## Local Testing

### Test CI Workflow Locally
You can test the CI workflow locally by running:

```bash
uv sync --extra torch-cpu
uv run pytest tests -v
```

### Build Docker Images Locally

```bash
# Build with torch-gpu backend (default)
docker build -t mussel:local-torch-gpu .

# Build with different backend
docker build --build-arg BACKEND=torch-cpu -t mussel:local-torch-cpu .
docker build --build-arg BACKEND=tensorflow-gpu -t mussel:local-tensorflow-gpu .
docker build --build-arg BACKEND=tensorflow-cpu -t mussel:local-tensorflow-cpu .
```

## Workflow Customization

### Changing Python Versions
Edit `.github/workflows/ci.yml` and modify the `matrix.python-version` array to add or remove Python versions.

### Changing Backends
Edit `.github/workflows/docker.yml` and modify the `matrix.backend` array to add or remove backend configurations.

### Adding More Triggers
Both workflows can be triggered manually using the "Actions" tab in GitHub (workflow_dispatch is enabled for the Docker workflow).
