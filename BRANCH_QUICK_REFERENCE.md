# PR Separation Quick Reference

## Branch List

This document provides a quick reference for the separated branches.

### Feature Branches (To Be Created)

The following 8 branches will be created when you run `./create-and-push-branches.sh`:

1. `feature/01-project-setup` - Project setup and configuration (6 files)
2. `feature/02-documentation` - Project documentation (10 files)
3. `feature/03-core-application` - Core application code and CLI (28 files)
4. `feature/04-presets` - Configuration presets (3 files)
5. `feature/05-docker-support` - Docker support (4 files)
6. `feature/06-ci-cd` - CI/CD workflows (3 files)
7. `feature/07-tests-code` - Test code (15 files)
8. `feature/08-tests-data` - Test data (8 files)

**Total: 77 files**

## Merge Order

### Phase 1 (Base)
- ✅ `feature/01-project-setup` ← **MERGE FIRST**

### Phase 2 (Independent branches, can merge in any order)
- `feature/02-documentation`
- `feature/04-presets`
- `feature/08-tests-data`

### Phase 3 (Requires Phase 1 + Phase 2)
- ✅ `feature/03-core-application` ← **MERGE BEFORE PHASE 4**

### Phase 4 (Independent branches, can merge in any order, requires Phase 3)
- `feature/05-docker-support`
- `feature/06-ci-cd`
- `feature/07-tests-code`

## Quick Commands

### View a specific branch
```bash
git checkout feature/01-project-setup
```

### List files in a branch
```bash
git checkout feature/01-project-setup
git ls-files
```

### Compare branch with base
```bash
git diff --stat 91c4977 feature/01-project-setup
```

### Create PR from branch
Each branch should have its own PR created from it, with the target being the main branch after the dependencies are merged.

## Dependencies Graph

```
project-setup (1)
    ├── documentation (2)
    ├── presets (4)
    ├── tests-data (8)
    └── core-application (3)
            ├── docker-support (5)
            ├── ci-cd (6)
            └── tests-code (7)
```

## Review Checklist

- [ ] feature/01-project-setup (6 files)
- [ ] feature/02-documentation (10 files)
- [ ] feature/04-presets (3 files)
- [ ] feature/08-tests-data (8 files)
- [ ] feature/03-core-application (28 files)
- [ ] feature/05-docker-support (4 files)
- [ ] feature/06-ci-cd (3 files)
- [ ] feature/07-tests-code (15 files)
