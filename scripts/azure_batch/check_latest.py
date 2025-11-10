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

job_id = "mussel-test-20251108-213440"
tasks = list(batch_client.task.list(job_id))
success_count = sum(1 for t in tasks if t.execution_info and t.execution_info.exit_code == 0)

print(f"Result: {success_count}/{len(tasks)} tasks succeeded")

if success_count == len(tasks):
    print("✅✅✅ ALL TASKS SUCCESSFUL! ✅✅✅")
else:
    for task in tasks:
        if task.execution_info and task.execution_info.exit_code != 0:
            print(f"❌ {task.id}: exit code {task.execution_info.exit_code}")
