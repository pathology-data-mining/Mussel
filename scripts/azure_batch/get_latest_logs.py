#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '../common')
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
import yaml

with open("../../azure_test.yaml", 'r') as f:
    config = yaml.safe_load(f)

account = config['azure']['batch_account_name']
url = config['azure']['batch_account_url']
key = os.environ.get("AZURE_BATCH_ACCOUNT_KEY")

credentials = SharedKeyCredentials(account, key)
batch_client = BatchServiceClient(credentials, batch_url=url)

job_id = "mussel-test-20251108-212042"
tasks = list(batch_client.task.list(job_id))

# Just check first task
task = tasks[0]
print(f"Task: {task.id}")
print(f"Exit code: {task.execution_info.exit_code}")

try:
    stderr_stream = batch_client.file.get_from_task(job_id, task.id, "stderr.txt")
    stderr = b''.join(stderr_stream).decode('utf-8')
    
    stdout_stream = batch_client.file.get_from_task(job_id, task.id, "stdout.txt")
    stdout = b''.join(stdout_stream).decode('utf-8')
    
    print("\nSTDERR (last 1500 chars):")
    print(stderr[-1500:])
    
    print("\nSTDOUT (last 1500 chars):")
    print(stdout[-1500:])
except Exception as e:
    print(f"Error: {e}")
