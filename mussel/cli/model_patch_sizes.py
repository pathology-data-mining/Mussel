"""Print each model's recommended tile size (and target MPP, where defined) as JSON.

Pipelines use this to tile every model at its recommended size; e.g. mussel-nf
generates and checks its copy of the table from this output.

Usage: model_patch_sizes            # {"patch_sizes": {"titan_slide": 512, ...}, "target_mpp": {...}}
"""
import json
import sys

from mussel.models import MODEL_TARGET_MPP, recommended_patch_sizes


def main() -> None:
    json.dump(
        {
            "patch_sizes": recommended_patch_sizes(),
            "target_mpp": {m.code: mpp for m, mpp in MODEL_TARGET_MPP.items()},
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
