# Branch Separation Plan for Mussel Project

## Overview
The original PR (commit 91c4977) contains 77 files with 12,474+ insertions, representing the entire initial project. This has been separated into 8 logical, manageable branches for easier review and incremental merging.

## Branch Strategy

### Dependency Order
Branches should be merged in the following order to maintain dependencies:

1. **Branch 1: project-setup** - Project infrastructure (MUST BE FIRST)
2. **Branch 2: documentation** - Project documentation
3. **Branch 3: core-application** - Core application code, utilities, and CLI commands
4. **Branch 4: presets** - Configuration presets
5. **Branch 5: docker-support** - Docker configuration and scripts
6. **Branch 6: ci-cd** - GitHub Actions workflows
7. **Branch 7: tests-code** - Test files (without test data)
8. **Branch 8: tests-data** - Test data files

## Detailed Branch Contents

### Branch 1: project-setup (6 files)
**Branch name:** `feature/01-project-setup`  
**Base:** Empty repository or grafted base  
**Purpose:** Essential project configuration files  
**Files:**
- pyproject.toml
- uv.lock
- .gitignore
- .dockerignore
- Makefile
- LICENSE.md

**Rationale:** These files define the project structure, dependencies, and build system. They must be present before any code.

### Branch 2: documentation (10 files)
**Branch name:** `feature/02-documentation`  
**Base:** project-setup  
**Purpose:** Project documentation  
**Files:**
- README.md
- README-commands.md
- README-docker.md
- CHANGELOG.md
- CONTRIBUTING.md
- docs/example-browse.png
- docs/example-browse2.png
- docs/example-interrog.png
- docs/example-mask.jpg
- docs/mussel.jpg

**Rationale:** Documentation can be reviewed independently and helps reviewers understand the project.

### Branch 3: core-application (28 files)
**Branch name:** `feature/03-core-application`  
**Base:** project-setup  
**Purpose:** Complete core application code including datasets, models, utilities, and CLI commands  
**Files:**
- mussel/__init__.py
- mussel/datasets/__init__.py
- mussel/datasets/h5.py
- mussel/datasets/tile_coords.py
- mussel/datasets/utils.py
- mussel/models/__init__.py
- mussel/models/model_factory.py
- mussel/models/resnet_custom.py
- mussel/utils/__init__.py
- mussel/utils/feature_extract.py
- mussel/utils/file.py
- mussel/utils/ml.py
- mussel/utils/reef.py
- mussel/utils/segment.py
- mussel/utils/tile_export.py
- mussel/utils/timer.py
- mussel/utils/wsi_classes.py
- mussel/cli/__init__.py
- mussel/cli/annotate.py
- mussel/cli/cache_tiles.py
- mussel/cli/create_class_embeddings.py
- mussel/cli/export_tiles.py
- mussel/cli/extract_features.py
- mussel/cli/filter_features.py
- mussel/cli/linear_probe_benchmark.py
- mussel/cli/merge_annotation_features.py
- mussel/cli/save_model.py
- mussel/cli/tessellate.py

**Rationale:** Grouping all core application code together makes for a more cohesive review of the application's functionality. This includes the data structures, models, utilities, and command-line interface as a single unit.

### Branch 4: presets (3 files)
**Branch name:** `feature/04-presets`  
**Base:** project-setup  
**Purpose:** Configuration presets  
**Files:**
- presets/bwh_biopsy.csv
- presets/bwh_resection.csv
- presets/tcga.csv

**Rationale:** Simple configuration files that don't depend on code.

### Branch 5: docker-support (4 files)
**Branch name:** `feature/05-docker-support`  
**Base:** core-application  
**Purpose:** Docker containerization  
**Files:**
- Dockerfile
- mussel-docker
- docker-example.sh
- test-docker-wrapper.sh

**Rationale:** Docker setup needs the application code to be present.

### Branch 6: ci-cd (3 files)
**Branch name:** `feature/06-ci-cd`  
**Base:** core-application  
**Purpose:** Continuous Integration and Deployment  
**Files:**
- .github/GITHUB_ACTIONS.md
- .github/workflows/ci.yml
- .github/workflows/docker.yml

**Rationale:** CI/CD workflows test the application, so they need the code first.

### Branch 7: tests-code (15 files)
**Branch name:** `feature/07-tests-code`  
**Base:** core-application  
**Purpose:** Test code  
**Files:**
- tests/mussel/cli/test_annotate.py
- tests/mussel/cli/test_cache_tiles.py
- tests/mussel/cli/test_create_class_embeddings.py
- tests/mussel/cli/test_extract_features.py
- tests/mussel/cli/test_filter_features.py
- tests/mussel/cli/test_tessellate.py
- tests/mussel/datasets/test_h5.py
- tests/mussel/models/test_model_factory.py
- tests/mussel/models/test_resnet_custom.py
- tests/mussel/test_utils.py
- tests/mussel/utils/test_file.py
- tests/mussel/utils/test_ml.py
- tests/mussel/utils/test_segment.py
- tests/mussel/utils/test_timer.py
- tests/mussel/utils/test_wsi_classes.py

**Rationale:** Tests verify the application code.

### Branch 8: tests-data (8 files)
**Branch name:** `feature/08-tests-data`  
**Base:** project-setup  
**Purpose:** Test data files (large binary files)  
**Files:**
- tests/testdata/948176.annotation.csv
- tests/testdata/948176.features.h5
- tests/testdata/948176.features.pt
- tests/testdata/948176.indices.json
- tests/testdata/948176.patch.h5
- tests/testdata/948176.svs
- tests/testdata/class_embedding.pt
- tests/testdata/simple_classifier.pkl

**Rationale:** Large binary test data should be separated for easier review and to avoid bloating early branches.

## Merge Order

```
Empty/Grafted Base
    ↓
project-setup ─────────┬─────────────────────────┐
    ↓                  ↓                         ↓
documentation      presets                  tests-data
    ↓                  ↓                         ↓
core-application ←─────┴─────────────────────────┤
    ↓                                            │
docker-support                                   │
    ↓                                            │
ci-cd                                            │
    ↓                                            │
tests-code ←─────────────────────────────────────┘
```

## How to Use These Branches

### For Reviewers
1. Review and approve branches in the order specified above
2. Branches can be reviewed in parallel where dependencies allow:
   - `documentation`, `presets`, and `tests-data` can all be reviewed in parallel after `project-setup`
   - `docker-support`, `ci-cd`, and `tests-code` can be reviewed in parallel after `core-application`

### For Merging
1. Merge `feature/01-project-setup` first (required for everything else)
2. Merge `feature/02-documentation`, `feature/04-presets`, and `feature/08-tests-data` (can be done in any order)
3. Merge `feature/03-core-application` (requires project-setup, presets)
4. Merge `feature/05-docker-support`, `feature/06-ci-cd`, and `feature/07-tests-code` (can be done in any order, all require core-application)

### Viewing Branch Contents
To see what files are in each branch:
```bash
git checkout feature/01-project-setup
git ls-files
```

### Checking Out All Branches Locally
```bash
git fetch origin
git checkout feature/01-project-setup
git checkout feature/02-documentation
git checkout feature/03-core-application
git checkout feature/04-presets
git checkout feature/05-docker-support
git checkout feature/06-ci-cd
git checkout feature/07-tests-code
git checkout feature/08-tests-data
```

## Summary Statistics

| Branch | Files | Description |
|--------|-------|-------------|
| feature/01-project-setup | 6 | Project configuration and build files |
| feature/02-documentation | 10 | Documentation and images |
| feature/03-core-application | 28 | Core code: datasets, models, utils, CLI |
| feature/04-presets | 3 | Configuration presets |
| feature/05-docker-support | 4 | Docker files and scripts |
| feature/06-ci-cd | 3 | GitHub Actions workflows |
| feature/07-tests-code | 15 | Test code |
| feature/08-tests-data | 8 | Test data files |
| **Total** | **77** | **All files from original PR** |

## Benefits of This Approach

1. **Easier Review:** Smaller, focused PRs are easier to review thoroughly
2. **Incremental Merge:** Can merge foundational pieces first and build on them
3. **Parallel Review:** Independent branches can be reviewed simultaneously by different reviewers
4. **Clear Dependencies:** Explicit dependency chain prevents merge conflicts
5. **Better Git History:** Clean, logical progression of changes in the repository
6. **Reduced Risk:** Issues in one branch don't block others
7. **Flexibility:** Can merge branches as they're approved without waiting for everything
