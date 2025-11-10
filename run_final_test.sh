#!/bin/bash
cd scripts/azure_batch
source ../../secrets.env
uv run python submit_batch_jobs.py \
  --config ../../azure_test.yaml \
  --csv-manifest ../../test_slides_quick.csv \
  --job-id mussel-patch-fix-test \
  --monitor
