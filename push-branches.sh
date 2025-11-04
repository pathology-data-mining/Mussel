#!/bin/bash
# Script to push all feature branches to remote repository
# Run this script to make all the separated branches available on GitHub

set -e

REPO_DIR="/home/runner/work/Mussel/Mussel"
cd "$REPO_DIR"

echo "Pushing all feature branches to origin..."
echo ""

BRANCHES=(
    "feature/01-project-setup"
    "feature/02-documentation"
    "feature/03-core-application"
    "feature/04-presets"
    "feature/05-docker-support"
    "feature/06-ci-cd"
    "feature/07-tests-code"
    "feature/08-tests-data"
)

for branch in "${BRANCHES[@]}"; do
    echo "Pushing $branch..."
    git push -u origin "$branch"
    echo ""
done

echo "All feature branches have been pushed successfully!"
echo ""
echo "You can now create pull requests for each branch in this order:"
echo "  1. feature/01-project-setup (FIRST)"
echo "  2. feature/02-documentation, feature/04-presets, feature/08-tests-data (parallel)"
echo "  3. feature/03-core-application"
echo "  4. feature/05-docker-support, feature/06-ci-cd, feature/07-tests-code (parallel)"
