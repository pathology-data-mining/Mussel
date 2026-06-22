import subprocess
import sys


def run_clean_import(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_extract_features_cli_imports_in_clean_interpreter():
    run_clean_import("import mussel.cli.extract_features")


def test_extract_features_cli_imports_after_datasets():
    run_clean_import(
        "import mussel.datasets; "
        "import mussel.cli.extract_features; "
        "from mussel.utils import aggregate_slide_features_batch"
    )


def test_feature_extract_exports_are_lazy_importable():
    run_clean_import(
        "from mussel.utils import ("
        "DatasetProcessor, FeatureExtractionResult, aggregate_slide_features_batch"
        ")"
    )
