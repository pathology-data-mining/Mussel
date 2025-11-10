import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

batch_account_name = os.environ['AZURE_BATCH_ACCOUNT_NAME']
batch_account_key = os.environ['AZURE_BATCH_ACCOUNT_KEY']
batch_account_url = os.environ['AZURE_BATCH_ACCOUNT_URL']

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_account_url)

job_id = "mussel-test-job"
tasks = list(batch_client.task.list(job_id))

for task in tasks:
    print(f"\n{'='*60}")
    print(f"Task: {task.id}")
    print(f"State: {task.state}")
    print(f"Exit code: {task.execution_info.exit_code if task.execution_info else 'N/A'}")
    
    if task.execution_info and task.execution_info.failure_info:
        print(f"Failure category: {task.execution_info.failure_info.category}")
        print(f"Failure code: {task.execution_info.failure_info.code}")
        print(f"Failure message: {task.execution_info.failure_info.message}")
        
        if task.execution_info.failure_info.details:
            print("Failure details:")
            for detail in task.execution_info.failure_info.details:
                print(f"  - {detail.name}: {detail.value}")
    
    if task.execution_info:
        print(f"Start time: {task.execution_info.start_time}")
        print(f"End time: {task.execution_info.end_time}")
        if task.execution_info.start_time and task.execution_info.end_time:
            duration = task.execution_info.end_time - task.execution_info.start_time
            print(f"Duration: {duration}")
