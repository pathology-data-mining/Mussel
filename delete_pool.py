import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

batch_account_name = "ocra"
batch_account_key = os.environ['AZURE_BATCH_ACCOUNT_KEY']
batch_account_url = "https://ocra.eastus2.batch.azure.com"

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_account_url)

pool_id = "mussel-pool"
job_id = "mussel-test-job"

print(f"Deleting job {job_id}...")
try:
    batch_client.job.delete(job_id)
    print(f"Job {job_id} deleted successfully")
except Exception as e:
    print(f"Error deleting job: {e}")

print(f"\nDeleting pool {pool_id}...")
try:
    batch_client.pool.delete(pool_id)
    print(f"Pool {pool_id} deleted successfully")
except Exception as e:
    print(f"Error deleting pool: {e}")
