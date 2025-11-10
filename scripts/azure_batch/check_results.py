#!/usr/bin/env python3
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
from azure.storage.fileshare import ShareServiceClient
import os
import sys
import yaml

# Load config for defaults
config_path = '../../azure_test.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Setup Batch client
batch_account_name = os.environ.get('AZURE_BATCH_ACCOUNT_NAME', config['azure']['batch_account_name'])
batch_account_key = os.environ.get('AZURE_BATCH_ACCOUNT_KEY')
batch_account_url = os.environ.get('AZURE_BATCH_ACCOUNT_URL', config['azure']['batch_account_url'])

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_url=batch_account_url)

# Get task details
job_id = sys.argv[1] if len(sys.argv) > 1 else 'mussel-a100-test-final'
tasks = list(batch_client.task.list(job_id))

print(f'\n=== Task Results for {job_id} ===\n')
for task in tasks:
    print(f'Task: {task.id}')
    print(f'  State: {task.state}')
    print(f'  Exit code: {task.execution_info.exit_code if task.execution_info else None}')
    print(f'  Result: {task.execution_info.result if task.execution_info else None}')
    
    # Get stdout
    try:
        stdout_file = batch_client.file.get_from_task(job_id, task.id, 'stdout.txt')
        stdout_content = b''.join(stdout_file).decode('utf-8')
        lines = stdout_content.split('\n')
        print(f'  Stdout (last 30 lines):')
        for line in lines[-30:]:
            print(f'    {line}')
    except Exception as e:
        print(f'  Stdout: Error reading - {e}')
    
    # Get stderr  
    try:
        stderr_file = batch_client.file.get_from_task(job_id, task.id, 'stderr.txt')
        stderr_content = b''.join(stderr_file).decode('utf-8')
        if stderr_content.strip():
            lines = stderr_content.split('\n')
            print(f'  Stderr (last 30 lines):')
            for line in lines[-30:]:
                print(f'    {line}')
    except Exception as e:
        print(f'  Stderr: Error reading - {e}')
    
    print()

# Check Azure Files for outputs
print('\n=== Azure Files - outputs directory ===\n')
try:
    storage_account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME', config['azure']['storage_account_name'])
    storage_account_key = os.environ.get('AZURE_STORAGE_ACCOUNT_KEY')
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
    service_client = ShareServiceClient.from_connection_string(connection_string)
    share_client = service_client.get_share_client('mussel-staging')
    
    dir_client = share_client.get_directory_client('outputs')
    files_and_dirs = list(dir_client.list_directories_and_files())
    
    if not files_and_dirs:
        print('No output files found')
    else:
        for item in files_and_dirs:
            if item.is_directory:
                print(f'DIR: {item.name}/')
                # List contents
                subdir = dir_client.get_subdirectory_client(item.name)
                for subitem in subdir.list_directories_and_files():
                    print(f'  - {subitem.name}')
            else:
                print(f'FILE: {item.name}')
except Exception as e:
    print(f'Error accessing outputs: {e}')
