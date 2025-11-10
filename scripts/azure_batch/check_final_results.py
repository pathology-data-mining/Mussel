#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '../common')
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
import yaml

# Load config
with open("../../azure_test.yaml", 'r') as f:
    config = yaml.safe_load(f)

account = config['azure']['batch_account_name']
url = config['azure']['batch_account_url']
key = os.environ.get("AZURE_BATCH_ACCOUNT_KEY")

credentials = SharedKeyCredentials(account, key)
batch_client = BatchServiceClient(credentials, batch_url=url)

job_id = "mussel-test-20251108-212042"
print(f"Checking job: {job_id}")
print("=" * 80)

tasks = list(batch_client.task.list(job_id))
success_count = 0
failure_count = 0

for task in tasks:
    if task.execution_info and task.execution_info.exit_code == 0:
        success_count += 1
        print(f"✅ {task.id}: SUCCESS (exit code 0)")
    else:
        failure_count += 1
        exit_code = task.execution_info.exit_code if task.execution_info else "Unknown"
        print(f"❌ {task.id}: FAILED (exit code {exit_code})")

print("\n" + "=" * 80)
print(f"Summary: {success_count} succeeded, {failure_count} failed out of {len(tasks)} total")
