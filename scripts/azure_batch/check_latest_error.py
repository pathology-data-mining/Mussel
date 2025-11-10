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

stderr = b''.join(batch_client.file.get_from_task(job_id, task.id, "stderr.txt")).decode('utf-8')
stdout = b''.join(batch_client.file.get_from_task(job_id, task.id, "stdout.txt")).decode('utf-8')

print("Exit code:", task.execution_info.exit_code)
print("\nLast 40 lines of STDERR:")
print('\n'.join(stderr.split('\n')[-40:]))
print("\n\nChecking for model path in STDOUT:")
for line in stdout.split('\n'):
    if 'PREFILTER_MODEL_PATH' in line or 'model_path' in line.lower():
        print(line)
