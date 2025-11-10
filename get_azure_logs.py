import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

batch_account_name = "ocra"
batch_account_key = os.environ['AZURE_BATCH_ACCOUNT_KEY']
batch_account_url = "https://ocra.eastus2.batch.azure.com"

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_account_url)

job_id = "mussel-test-job"
task_id = "1079807"

print(f"\n=== Task {task_id} Logs ===\n")

# Get stdout
try:
    stdout = batch_client.file.get_from_task(job_id, task_id, "stdout.txt")
    print("--- STDOUT ---")
    for chunk in stdout:
        print(chunk.decode('utf-8'), end='')
    print()
except Exception as e:
    print(f"Could not get stdout: {e}")

# Get stderr
try:
    stderr = batch_client.file.get_from_task(job_id, task_id, "stderr.txt")
    print("\n--- STDERR ---")
    for chunk in stderr:
        print(chunk.decode('utf-8'), end='')
    print()
except Exception as e:
    print(f"Could not get stderr: {e}")

# List all files
try:
    print("\n--- Available Files ---")
    files = batch_client.file.list_from_task(job_id, task_id)
    for f in files:
        if not f.is_directory:
            print(f"  {f.name} ({f.properties.content_length} bytes)")
except Exception as e:
    print(f"Could not list files: {e}")
