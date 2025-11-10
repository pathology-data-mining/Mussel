#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '../common')
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

# Get credentials
account = os.environ["AZURE_BATCH_ACCOUNT_NAME"]
key = os.environ["AZURE_BATCH_ACCOUNT_KEY"]
url = os.environ["AZURE_BATCH_ACCOUNT_URL"]

# Create client
credentials = SharedKeyCredentials(account, key)
batch_client = BatchServiceClient(credentials, batch_url=url)

# Get tasks
job_id = "mussel-test-20251108-211053"
tasks = list(batch_client.task.list(job_id))

print(f"Tasks in job {job_id}:")
print("=" * 80)
for task in tasks:
    print(f"\nTask ID: {task.id}")
    print(f"  State: {task.state}")
    if task.execution_info:
        print(f"  Exit code: {task.execution_info.exit_code}")
        print(f"  Result: {task.execution_info.result}")
        if task.execution_info.failure_info:
            print(f"  Failure category: {task.execution_info.failure_info.category}")
            print(f"  Failure message: {task.execution_info.failure_info.message[:500]}")
        
        # Get stdout/stderr
        print(f"\n  Fetching task logs...")
        try:
            stdout = batch_client.file.get_from_task(job_id, task.id, "stdout.txt")
            stderr = batch_client.file.get_from_task(job_id, task.id, "stderr.txt")
            
            stdout_content = stdout.content.decode('utf-8')
            stderr_content = stderr.content.decode('utf-8')
            
            if stdout_content:
                print(f"\n  STDOUT (last 1000 chars):")
                print("  " + "\n  ".join(stdout_content[-1000:].split('\n')))
            
            if stderr_content:
                print(f"\n  STDERR (last 1000 chars):")
                print("  " + "\n  ".join(stderr_content[-1000:].split('\n')))
        except Exception as e:
            print(f"  Could not fetch logs: {e}")
