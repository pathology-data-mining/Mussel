import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

batch_account_name = "ocra"
batch_account_key = os.environ['AZURE_BATCH_ACCOUNT_KEY']
batch_account_url = "https://ocra.eastus2.batch.azure.com"

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_account_url)

job_id = "mussel-test-job"
tasks = list(batch_client.task.list(job_id))

for task in tasks:
    print(f"\n{'='*60}")
    print(f"Task {task.id} - State: {task.state}")
    print(f"{'='*60}")
    
    if task.execution_info:
        print(f"Exit code: {task.execution_info.exit_code}")
        if task.execution_info.failure_info:
            print(f"Failure: {task.execution_info.failure_info.message}")
    
    try:
        # Get stderr - use read method
        stderr_stream = batch_client.file.get_from_task(job_id, task.id, "stderr.txt")
        stderr_bytes = b''
        for chunk in stderr_stream:
            stderr_bytes += chunk
        stderr_text = stderr_bytes.decode('utf-8', errors='ignore')
        print(f"\n--- STDERR (last 3000 chars) ---")
        print(stderr_text[-3000:])
    except Exception as e:
        print(f"Could not get stderr: {e}")
    
    try:
        # Get stdout - use read method  
        stdout_stream = batch_client.file.get_from_task(job_id, task.id, "stdout.txt")
        stdout_bytes = b''
        for chunk in stdout_stream:
            stdout_bytes += chunk
        stdout_text = stdout_bytes.decode('utf-8', errors='ignore')
        print(f"\n--- STDOUT (last 1500 chars) ---")
        print(stdout_text[-1500:])
    except Exception as e:
        print(f"Could not get stdout: {e}")
