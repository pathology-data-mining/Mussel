#!/bin/bash -x

# Load credentials
if [ -f secrets.env ]; then
  source secrets.env
fi

export USE_AZCOPY=true
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

python scripts/azure_batch/submit_batch_jobs.py \
  --csv-manifest external_data_staged_manifest.csv \
  --config run_paper_revisions_prod.yaml \
  --pool-id mussel-external-data-pool \
  --create-pool \
  --create-job \
  --job-id external-data-$TIMESTAMP \
  --env-file secrets.env \
  --stage-to-azure-blob \
  --staging-workers 20 \
  --monitor \
  --save-failed-tasks external-data-failed-$TIMESTAMP.csv
