#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '../common')
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
import yaml

with open("../../azure_test.yaml", 'r') as f:
    config = yaml.safe_load(f)

credentials = SharedKeyCredentials(config['azure']['batch_account_name'], os.environ.get("AZURE_BATCH_ACCOUNT_KEY"))
batch_client = BatchServiceClient(credentials, batch_url=config['azure']['batch_account_url'])

pool_id = "mussel-pool"
print(f"Deleting pool: {pool_id}")
try:
    batch_client.pool.delete(pool_id)
    print(f"✓ Pool {pool_id} deleted successfully")
except Exception as e:
    print(f"Error: {e}")
