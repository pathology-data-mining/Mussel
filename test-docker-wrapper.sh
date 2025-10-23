#!/usr/bin/env bash
#
# Test script for mussel-docker wrapper
# This verifies that the wrapper script works correctly
#

set -e

echo "=== Testing mussel-docker wrapper ==="
echo ""

# Test 1: Help command
echo "Test 1: Help command"
output=$(./mussel-docker help 2>&1)
if echo "$output" | grep -q "Available Commands"; then
    echo "✓ Help command works"
else
    echo "✗ Help command failed"
    exit 1
fi

# Test 2: No arguments
echo "Test 2: No arguments (should show help)"
output=$(./mussel-docker 2>&1)
if echo "$output" | grep -q "Available Commands"; then
    echo "✓ No arguments shows help"
else
    echo "✗ No arguments test failed"
    exit 1
fi

# Test 3: Invalid command
echo "Test 3: Invalid command"
output=$(./mussel-docker invalid_command 2>&1 || true)
if echo "$output" | grep -q "Unknown command"; then
    echo "✓ Invalid command error handling works"
else
    echo "✗ Invalid command test failed"
    exit 1
fi

# Test 4: Check all commands are recognized
echo "Test 4: Verify all commands are recognized"
commands=(
    "tessellate"
    "extract_features"
    "create_class_embeddings"
    "annotate"
    "cache_tiles"
    "export_tiles"
    "filter_features"
    "merge_annotation_features"
    "linear_probe_benchmark"
    "save_model"
)

for cmd in "${commands[@]}"; do
    if ./mussel-docker help | grep -q "$cmd"; then
        echo "  ✓ $cmd command listed"
    else
        echo "  ✗ $cmd command not found in help"
        exit 1
    fi
done

# Test 5: Environment variable recognition
echo "Test 5: Environment variable recognition"
export MUSSEL_DOCKER_IMAGE="test-image:test"
export MUSSEL_BACKEND="torch-cpu"
export MUSSEL_USE_GPU="false"
# Just verify the script runs without error when these are set
./mussel-docker help > /dev/null 2>&1
echo "✓ Environment variables recognized"

# Test 6: Bash syntax
echo "Test 6: Bash syntax check"
bash -n ./mussel-docker
echo "✓ Bash syntax is valid"

bash -n ./docker-example.sh
echo "✓ docker-example.sh syntax is valid"

echo ""
echo "=== All tests passed! ==="
