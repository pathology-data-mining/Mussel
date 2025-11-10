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

job_id = "mussel-test-20251108-233519"
task = list(batch_client.task.list(job_id))[0]

stdout = b''.join(batch_client.file.get_from_task(job_id, task.id, "stdout.txt")).decode('utf-8')
stderr = b''.join(batch_client.file.get_from_task(job_id, task.id, "stderr.txt")).decode('utf-8')

print("Exit code:", task.execution_info.exit_code)
print("\nSearching for prefilter_model_path in command:")
for line in stdout.split('\n'):
    if 'prefilter_model_path=' in line:
        print(line)
        break

print("\nLast error:")
print('\n'.join(stderr.split('\n')[-15:]))
