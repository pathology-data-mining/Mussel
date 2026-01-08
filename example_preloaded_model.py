"""
Example demonstrating how to use get_features with a pre-loaded model.

This modification allows you to load a model once and reuse it across multiple
get_features calls, avoiding the overhead of reloading the model each time.
"""

from mussel.models import ModelType, get_model_factory
from mussel.utils import get_features

# Example 1: Traditional usage (backward compatible)
# The model will be loaded internally
def example_traditional(coords, slide_path, attrs):
    features, labels = get_features(
        coords=coords,
        slide_path=slide_path,
        attrs=attrs,
        model_type=ModelType.CLIP,
        model_path=None,
    )
    return features, labels

# Example 2: Using a pre-loaded model
# Load the model once and reuse it
def example_preloaded(coords_list, slide_paths, attrs_list):
    # Load the model once
    model_factory = get_model_factory(ModelType.CLIP)
    model = model_factory.get_model(
        model_path=None,
        use_gpu=True,
        gpu_device_id=0
    )

    # Reuse the model for multiple slides
    results = []
    for coords, slide_path, attrs in zip(coords_list, slide_paths, attrs_list):
        features, labels = get_features(
            coords=coords,
            slide_path=slide_path,
            attrs=attrs,
            model=model,  # Pass the pre-loaded model
        )
        results.append((features, labels))

    return results

# Example 3: Using a custom model
def example_custom_model(coords, slide_path, attrs, custom_model):
    """
    You can pass any Model instance that has the expected interface:
    - get_preprocessing_fun() method
    - get_model_fun() method
    """
    features, labels = get_features(
        coords=coords,
        slide_path=slide_path,
        attrs=attrs,
        model=custom_model,
    )
    return features, labels

if __name__ == "__main__":
    print("This is an example file demonstrating usage.")
    print("The get_features function now accepts an optional 'model' parameter.")
    print("\nBenefits:")
    print("- Backward compatible: existing code continues to work")
    print("- Performance: avoid reloading models for batch processing")
    print("- Flexibility: use custom model instances")
