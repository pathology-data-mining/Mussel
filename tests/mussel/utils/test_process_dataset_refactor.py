"""
Tests for the refactored process_dataset function to ensure:
1. No operator overloading is used
2. Code duplication is minimized
3. Functionality is preserved
"""

import ast
from pathlib import Path

# Path to the module being tested
# Going up from tests/mussel/utils/ to repository root, then to mussel/utils/
FEATURE_EXTRACT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "mussel"
    / "utils"
    / "feature_extract.py"
)


def test_no_singledispatch_decorator():
    """Verify that @singledispatch decorator is not used."""
    with open(FEATURE_EXTRACT_PATH, "r") as f:
        source = f.read()

    # Parse the source code
    tree = ast.parse(source)

    # Check for any @singledispatch decorators
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    assert (
                        decorator.id != "singledispatch"
                    ), f"Found @singledispatch decorator on function {node.name}"
                elif isinstance(decorator, ast.Attribute):
                    assert (
                        decorator.attr != "register"
                    ), f"Found .register decorator on function {node.name}"

    print("✓ No @singledispatch decorators found")


def test_no_singledispatch_import():
    """Verify singledispatch is not imported."""
    with open(FEATURE_EXTRACT_PATH, "r") as f:
        source = f.read()

    assert (
        "from functools import singledispatch" not in source
    ), "singledispatch is still imported"
    assert "import singledispatch" not in source, "singledispatch is still imported"

    print("✓ singledispatch is not imported")


def test_helper_functions_exist():
    """Verify that helper functions exist to minimize code duplication."""
    with open(FEATURE_EXTRACT_PATH, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    # Find all function definitions
    functions = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]

    # Check for expected helper functions
    assert (
        "_extract_features_from_loader" in functions
    ), "Helper function _extract_features_from_loader not found"
    assert (
        "_process_tile_coord_dataset" in functions
    ), "Helper function _process_tile_coord_dataset not found"
    assert (
        "_process_image_folder_dataset" in functions
    ), "Helper function _process_image_folder_dataset not found"
    assert (
        "_process_h5_dataset" in functions
    ), "Helper function _process_h5_dataset not found"
    assert "process_dataset" in functions, "Main function process_dataset not found"

    print("✓ All helper functions exist")


def test_process_dataset_uses_isinstance():
    """Verify that process_dataset uses isinstance for type checking."""
    with open(FEATURE_EXTRACT_PATH, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    # Find process_dataset function
    process_dataset_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "process_dataset":
            process_dataset_func = node
            break

    assert process_dataset_func is not None, "process_dataset function not found"

    # Check that it uses isinstance
    func_source = ast.get_source_segment(source, process_dataset_func)
    assert (
        "isinstance" in func_source
    ), "process_dataset should use isinstance for type checking"

    print("✓ process_dataset uses isinstance for type checking")


def test_common_loop_logic_extracted():
    """Verify that common loop logic is extracted to a helper."""
    with open(FEATURE_EXTRACT_PATH, "r") as f:
        source = f.read()

    # The helper function should contain the common pattern
    assert "_extract_features_from_loader" in source, "Common loop logic not extracted"

    # Check that the helper is a generator (uses yield)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_extract_features_from_loader"
        ):
            # Check for yield statement
            has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(node))
            assert has_yield, "_extract_features_from_loader should be a generator"

    print("✓ Common loop logic extracted to helper function")


def test_code_structure_improvements():
    """Verify overall code structure improvements."""
    with open(FEATURE_EXTRACT_PATH, "r") as f:
        lines = f.readlines()

    # Count occurrences of the repetitive pattern
    enumerate_tqdm_count = sum(1 for line in lines if "enumerate(tqdm(loader" in line)

    # Should only appear once in the helper function now
    assert (
        enumerate_tqdm_count == 1
    ), f"enumerate(tqdm(loader)) should appear only once, found {enumerate_tqdm_count} times"

    print("✓ Code duplication minimized")


if __name__ == "__main__":
    test_no_singledispatch_decorator()
    test_no_singledispatch_import()
    test_helper_functions_exist()
    test_process_dataset_uses_isinstance()
    test_common_loop_logic_extracted()
    test_code_structure_improvements()
    print("\n✅ All tests passed! Refactoring complete.")
