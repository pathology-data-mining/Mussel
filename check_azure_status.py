import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

batch_account_name = "ocra"
batch_account_key = os.environ['AZURE_BATCH_ACCOUNT_KEY']
batch_account_url = "https://ocra.eastus2.batch.azure.com"

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_account_url)

job_id = "mussel-test-job"

print(f"\nChecking job: {job_id}")
tasks = list(batch_client.task.list(job_id))
print(f"Total tasks: {len(tasks)}\n")

for task in tasks:
    print(f"Task: {task.id}")
    print(f"  State: {task.state}")
    if task.execution_info:
        print(f"  Start time: {task.execution_info.start_time}")
        print(f"  End time: {task.execution_info.end_time}")
        if task.execution_info.exit_code is not None:
            print(f"  Exit code: {task.execution_info.exit_code}")
        if task.execution_info.failure_info:
            print(f"  Failure: {task.execution_info.failure_info.category} - {task.execution_info.failure_info.message}")
    print()
