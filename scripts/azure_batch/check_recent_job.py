#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '../common')
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
from azure.storage.fileshare import ShareServiceClient
import yaml

with open("../../azure_test.yaml", 'r') as f:
    config = yaml.safe_load(f)

credentials = SharedKeyCredentials(config['azure']['batch_account_name'], os.environ.get("AZURE_BATCH_ACCOUNT_KEY"))
batch_client = BatchServiceClient(credentials, batch_url=config['azure']['batch_account_url'])

# List recent jobs
jobs = list(batch_client.job.list())
recent_jobs = sorted(jobs, key=lambda j: j.creation_time, reverse=True)[:3]

print("Recent jobs:")
for job in recent_jobs:
    print(f"  {job.id} - created {job.creation_time}")

# Check the most recent one
latest_job = recent_jobs[0]
job_id = latest_job.id
print(f"\nChecking latest job: {job_id}\n")

tasks = list(batch_client.task.list(job_id))
success_count = sum(1 for t in tasks if t.execution_info and t.execution_info.exit_code == 0)

print(f"Task Results: {success_count}/{len(tasks)} succeeded\n")

for task in tasks:
    status = "✅" if task.execution_info and task.execution_info.exit_code == 0 else "❌"
    exit_code = task.execution_info.exit_code if task.execution_info else "?"
    print(f"{status} {task.id}: exit code {exit_code}")

if success_count == len(tasks):
    print("\n" + "=" * 80)
    print("✅✅✅ ALL TASKS SUCCESSFUL! ✅✅✅")
    
    # Check Azure Files for output
    print("\nChecking Azure Files for results...")
    storage_key = os.environ.get("AZURE_STORAGE_KEY")
    service_client = ShareServiceClient(
        account_url="https://mskpdmgen2.file.core.windows.net",
        credential=storage_key
    )
    share_client = service_client.get_share_client("mussel-staging")
    
    try:
        dir_client = share_client.get_directory_client("outputs")
        files = list(dir_client.list_directories_and_files())
        print(f"\nFound {len(files)} items in outputs directory:")
        for item in files[:20]:
            if hasattr(item, 'size'):
                size_mb = item.size / (1024*1024)
                print(f"  {item.name} ({size_mb:.2f} MB)")
            else:
                print(f"  {item.name}/ (directory)")
    except Exception as e:
        print(f"Error listing outputs: {e}")
