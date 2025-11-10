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

job_id = "mussel-test-20251108-234714"
tasks = list(batch_client.task.list(job_id))

print(f"Job: {job_id}")
print(f"Total tasks: {len(tasks)}\n")

for task in tasks[:3]:
    print(f"Task: {task.id}")
    print(f"  State: {task.state}")
    if task.execution_info:
        print(f"  Exit code: {task.execution_info.exit_code}")
        print(f"  Result: {task.execution_info.result}")
        if task.execution_info.failure_info:
            print(f"  Failure: {task.execution_info.failure_info.message[:200]}")
    else:
        print(f"  No execution info yet")
    
    try:
        stderr = b''.join(batch_client.file.get_from_task(job_id, task.id, "stderr.txt")).decode('utf-8')
        if stderr:
            print(f"  STDERR (last 500 chars): {stderr[-500:]}")
        else:
            print(f"  No stderr")
    except Exception as e:
        print(f"  Error reading logs: {e}")
    print()
