#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '../common')
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

# Get credentials from config
import yaml
config_path = "../../azure_test.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

account = config['azure']['batch_account_name']
url = config['azure']['batch_account_url']
key = os.environ.get("AZURE_BATCH_ACCOUNT_KEY")

if not key:
    print("ERROR: AZURE_BATCH_ACCOUNT_KEY not set")
    sys.exit(1)

# Create client
credentials = SharedKeyCredentials(account, key)
batch_client = BatchServiceClient(credentials, batch_url=url)

# Get tasks
job_id = "mussel-test-20251108-211053"
print(f"Checking job: {job_id}")
print("=" * 80)

tasks = list(batch_client.task.list(job_id))
print(f"\nFound {len(tasks)} tasks\n")

for task in tasks:
    print(f"\nTask: {task.id}")
    print(f"  State: {task.state}")
    if task.execution_info:
        print(f"  Exit code: {task.execution_info.exit_code}")
        print(f"  Result: {task.execution_info.result}")
        
        if task.execution_info.failure_info:
            print(f"\n  FAILURE INFO:")
            print(f"    Category: {task.execution_info.failure_info.category}")
            print(f"    Code: {task.execution_info.failure_info.code}")
            print(f"    Message: {task.execution_info.failure_info.message}")
        
        # Get logs
        print(f"\n  Fetching logs...")
        try:
            stdout_stream = batch_client.file.get_from_task(job_id, task.id, "stdout.txt")
            stdout = b''.join(stdout_stream).decode('utf-8')
            
            stderr_stream = batch_client.file.get_from_task(job_id, task.id, "stderr.txt")
            stderr = b''.join(stderr_stream).decode('utf-8')
            
            if stderr:
                print(f"\n  STDERR (last 2000 chars):")
                print("  " + "\n  ".join(stderr[-2000:].split('\n')))
            
            if stdout:
                print(f"\n  STDOUT (last 2000 chars):")
                print("  " + "\n  ".join(stdout[-2000:].split('\n')))
            
        except Exception as e:
            print(f"  Error fetching logs: {e}")
    
    print("\n" + "-" * 80)
