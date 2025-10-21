# Contributing to Mussel

Thank you for your interest in contributing to Mussel! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

Before contributing, please:

1. Check existing [issues](https://github.com/pathology-data-mining/Mussel/issues) to avoid duplicate work
2. For major changes, open an issue first to discuss your proposed changes
3. Ensure you understand the project's [license](LICENSE.md) (GPL v3)

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/Mussel.git
cd Mussel
```

### 2. Install Dependencies

Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a development environment (choose based on your needs):
```bash
# For PyTorch with GPU
uv sync --extra torch-gpu

# For PyTorch CPU-only
uv sync --extra torch-cpu

# For TensorFlow with GPU
uv sync --extra tensorflow-gpu

# For TensorFlow CPU-only
uv sync --extra tensorflow-cpu
```

### 3. Activate the Environment

```bash
source .venv/bin/activate
```

## Making Changes

### Branch Naming

Create a descriptive branch for your changes:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### Code Organization

- **CLI tools**: `mussel/cli/` - Command-line interface scripts
- **Models**: `mussel/models/` - Model definitions and factory
- **Utils**: `mussel/utils/` - Utility functions
- **Datasets**: `mussel/datasets/` - Dataset classes and data loading
- **Tests**: `tests/` - Unit and integration tests

## Testing

### Running Tests

Always run tests before submitting changes:

```bash
# Run all tests
uv run pytest tests

# Run specific test file
uv run pytest tests/test_specific.py

# Run with verbose output
uv run pytest -v tests

# Run with coverage
uv run pytest --cov=mussel tests
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Include both positive and negative test cases
- Test edge cases and error handling

Example test structure:
```python
def test_feature_extraction_basic():
    """Test basic feature extraction functionality."""
    # Setup
    slide_path = "tests/testdata/sample.svs"
    
    # Execute
    result = extract_features(slide_path, ...)
    
    # Assert
    assert result is not None
    assert result.shape[0] > 0
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters maximum
- **Imports**: Group standard library, third-party, and local imports separately
- **Type hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings

### Formatting

Format your code using the development tools:

```bash
# Format code with black
uv run black mussel tests

# Sort imports with isort
uv run isort mussel tests

# Type checking with mypy
uv run mypy mussel
```

### Docstring Example

```python
def extract_features(slide_path: str, model_type: ModelType) -> np.ndarray:
    """Extract features from a whole-slide image.
    
    Args:
        slide_path: Path to the whole-slide image file.
        model_type: Type of foundation model to use.
        
    Returns:
        Array of feature embeddings with shape (n_tiles, n_features).
        
    Raises:
        FileNotFoundError: If slide_path does not exist.
        ValueError: If model_type is not supported.
    """
    # Implementation
    pass
```

## Submitting Changes

### 1. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description of changes

Detailed explanation of what changed and why.
Fixes #123"
```

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create a Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Provide a clear title and description
4. Reference any related issues
5. Wait for review and address feedback

### Pull Request Guidelines

- Keep changes focused and atomic
- Update documentation if needed
- Add or update tests for new features
- Ensure all tests pass
- Follow the code style guidelines
- Respond to review comments promptly

## Reporting Issues

### Before Opening an Issue

1. Search existing issues to avoid duplicates
2. Gather relevant information:
   - Operating system and version
   - Python version
   - Mussel version
   - Full error message and stack trace
   - Minimal reproducible example

### Issue Template

When opening an issue, include:

**Description**: Clear description of the problem or feature request

**Environment**:
- OS: [e.g., Ubuntu 22.04, macOS 13.0]
- Python version: [e.g., 3.11.5]
- Mussel version: [e.g., 1.0.1]
- Installation method: [e.g., uv sync --extra torch-gpu]

**Steps to Reproduce**:
1. Step one
2. Step two
3. Step three

**Expected Behavior**: What you expected to happen

**Actual Behavior**: What actually happened

**Error Messages**: Full error output (use code blocks)

**Additional Context**: Any other relevant information

## Adding New Models

If you're adding support for a new foundation model:

1. Add the model to `ModelType` enum in `mussel/models/model_factory.py`
2. Create a model class extending `Model` base class
3. Implement `get_model_fun()` and `get_preprocessing_fun()` methods
4. Add tests in `tests/test_models.py`
5. Update documentation:
   - README.md installation requirements
   - README-commands.md model table
6. Provide example usage

## Questions?

If you have questions about contributing:

- Open a [discussion](https://github.com/pathology-data-mining/Mussel/discussions)
- Check existing documentation
- Ask in an issue (if relevant to existing issue)

## Code of Conduct

Be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## License

By contributing to Mussel, you agree that your contributions will be licensed under the GPL v3 License.
