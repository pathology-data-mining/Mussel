#!/usr/bin/env python3
import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

# Load credentials from secrets.env
secrets = {}
with open('secrets.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            if line.startswith('export '):
                line = line[7:]
            key, value = line.split('=', 1)
            value = value.strip('"').strip("'")
            secrets[key] = value

batch_account_name = secrets.get('AZURE_BATCH_ACCOUNT_NAME')
batch_account_key = secrets.get('AZURE_BATCH_ACCOUNT_KEY')
batch_account_url = secrets.get('AZURE_BATCH_ACCOUNT_URL')

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_account_url)

# Get the most recent job
jobs = list(batch_client.job.list())
if jobs:
    job = sorted(jobs, key=lambda j: j.creation_time, reverse=True)[0]
    print(f"Job: {job.id}")
    print(f"State: {job.state}\n")
    
    # Get tasks
    tasks = list(batch_client.task.list(job.id))
    for task in tasks:
        print(f"Task: {task.id}")
        print(f"State: {task.state}")
        if task.execution_info:
            print(f"Exit code: {task.execution_info.exit_code}")
            if task.execution_info.failure_info:
                print(f"Failure: {task.execution_info.failure_info.category} - {task.execution_info.failure_info.message}")
        print()
