#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '../common')
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
import yaml

with open("../../azure_test.yaml", 'r') as f:
    config = yaml.safe_load(f)

credentials = SharedKeyCredentials(config['azure']['batch_account_name'], os.environ.get("AZURE_BATCH_ACCOUNT_KEY"))
batch_client = BatchServiceClient(credentials, batch_url=config['azure']['batch_account_url'])

job_id = "mussel-test-20251108-215530"
task = list(batch_client.task.list(job_id))[0]

stdout = b''.join(batch_client.file.get_from_task(job_id, task.id, "stdout.txt")).decode('utf-8')

print("Environment variables set in task:")
for line in stdout.split('\n')[:50]:
    if 'PREFILTER\|MODEL\|OUTPUT' in line or any(x in line for x in ['PREFILTER', 'MODEL', 'PATH']):
        print(line)
