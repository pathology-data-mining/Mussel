import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

batch_account_name = os.getenv("AZURE_BATCH_ACCOUNT_NAME", "ocra")
batch_account_key = os.getenv("AZURE_BATCH_ACCOUNT_KEY")
batch_account_url = os.getenv("AZURE_BATCH_ACCOUNT_URL", "https://ocra.eastus2.batch.azure.com")
job_id = "mussel-final-pr-job"

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_url=batch_account_url)

print(f"Deleting job: {job_id}")
try:
    batch_client.job.delete(job_id)
    print("Job deleted successfully")
except Exception as e:
    print(f"Error: {e}")
