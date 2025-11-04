#!/bin/bash
# Script to create and push all feature branches to remote repository
# This script creates the 8 separated branches from the original commit and pushes them to GitHub

set -e

REPO_DIR="/home/runner/work/Mussel/Mussel"
BASE_COMMIT="91c4977"  # The grafted commit with all files

cd "$REPO_DIR"

echo "=========================================="
echo "Creating and Pushing Feature Branches"
echo "=========================================="
echo ""

# Function to create a branch with specific files
create_and_push_branch() {
    local branch_name=$1
    local commit_msg=$2
    shift 2
    local files=("$@")
    
    echo "Creating branch: $branch_name"
    git checkout -b "$branch_name" "$BASE_COMMIT" 2>/dev/null || git checkout "$branch_name"
    
    # Remove all files from staging
    git rm -rf . >/dev/null 2>&1 || true
    
    # Checkout only the files we want
    for file in "${files[@]}"; do
        if [[ "$file" == */ ]]; then
            # It's a directory
            mkdir -p "$file"
            git checkout "$BASE_COMMIT" -- "$file" 2>/dev/null || true
        else
            # It's a file
            git checkout "$BASE_COMMIT" -- "$file" 2>/dev/null || true
        fi
    done
    
    git add -A
    git commit -m "$commit_msg" --allow-empty
    
    echo "Pushing $branch_name to origin..."
    git push -u origin "$branch_name"
    echo "✓ $branch_name created and pushed"
    echo ""
}

# Branch 1: project-setup
create_and_push_branch "feature/01-project-setup" \
    "Add project setup and configuration

- Add pyproject.toml with project dependencies and configuration
- Add uv.lock for dependency locking
- Add .gitignore for repository file exclusions
- Add .dockerignore for Docker build exclusions
- Add Makefile for build automation
- Add LICENSE.md for project licensing

This is the foundation branch that must be merged first." \
    "pyproject.toml" "uv.lock" ".gitignore" ".dockerignore" "Makefile" "LICENSE.md"

# Branch 2: documentation
create_and_push_branch "feature/02-documentation" \
    "Add project documentation

- Add README.md with project overview
- Add README-commands.md with command usage documentation
- Add README-docker.md with Docker usage documentation
- Add CHANGELOG.md for version history
- Add CONTRIBUTING.md with contribution guidelines
- Add docs/ directory with example images

Dependencies: feature/01-project-setup" \
    "README.md" "README-commands.md" "README-docker.md" "CHANGELOG.md" "CONTRIBUTING.md" "docs/"

# Branch 3: core-application
create_and_push_branch "feature/03-core-application" \
    "Add core application code and CLI commands

Core modules:
- mussel/datasets: Data handling (h5, tile_coords, utils)
- mussel/models: Model definitions (model_factory, resnet_custom)
- mussel/utils: Utility functions (feature_extract, file, ml, reef, segment, tile_export, timer, wsi_classes)
- mussel/cli: Command-line interface (annotate, cache_tiles, create_class_embeddings, export_tiles, extract_features, filter_features, linear_probe_benchmark, merge_annotation_features, save_model, tessellate)

Dependencies: feature/01-project-setup" \
    "mussel/"

# Branch 4: presets
create_and_push_branch "feature/04-presets" \
    "Add configuration presets

- Add presets/bwh_biopsy.csv
- Add presets/bwh_resection.csv
- Add presets/tcga.csv

Dependencies: feature/01-project-setup" \
    "presets/"

# Branch 5: docker-support
create_and_push_branch "feature/05-docker-support" \
    "Add Docker support

- Add Dockerfile for containerization
- Add mussel-docker wrapper script
- Add docker-example.sh for usage examples
- Add test-docker-wrapper.sh for testing

Dependencies: feature/03-core-application" \
    "Dockerfile" "mussel-docker" "docker-example.sh" "test-docker-wrapper.sh"

# Branch 6: ci-cd
create_and_push_branch "feature/06-ci-cd" \
    "Add CI/CD workflows

- Add .github/workflows/ci.yml for continuous integration
- Add .github/workflows/docker.yml for Docker builds
- Add .github/GITHUB_ACTIONS.md for workflow documentation

Dependencies: feature/03-core-application" \
    ".github/"

# Branch 7: tests-code
create_and_push_branch "feature/07-tests-code" \
    "Add test code

Test files:
- tests/mussel/cli/: CLI command tests
- tests/mussel/datasets/: Dataset tests
- tests/mussel/models/: Model tests
- tests/mussel/utils/: Utility function tests
- tests/test_utils.py: General test utilities

Dependencies: feature/03-core-application" \
    "tests/mussel/"

# Branch 8: tests-data
create_and_push_branch "feature/08-tests-data" \
    "Add test data files

Large binary test data files:
- tests/testdata/948176.* (various formats)
- tests/testdata/class_embedding.pt
- tests/testdata/simple_classifier.pkl

Dependencies: feature/01-project-setup" \
    "tests/testdata/"

# Return to original branch
git checkout copilot/separate-pr-into-branches

echo "=========================================="
echo "All branches created and pushed!"
echo "=========================================="
echo ""
echo "You can now create pull requests for each branch in this order:"
echo "  Phase 1: feature/01-project-setup (FIRST)"
echo "  Phase 2: feature/02-documentation, feature/04-presets, feature/08-tests-data (parallel)"
echo "  Phase 3: feature/03-core-application"
echo "  Phase 4: feature/05-docker-support, feature/06-ci-cd, feature/07-tests-code (parallel)"
