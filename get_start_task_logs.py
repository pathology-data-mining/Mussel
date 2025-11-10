import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

batch_account_name = "ocra"
batch_account_key = os.environ['AZURE_BATCH_ACCOUNT_KEY']
batch_account_url = "https://ocra.eastus2.batch.azure.com"

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_account_url)

pool_id = "mussel-pool"

# Get first node
nodes = list(batch_client.compute_node.list(pool_id))
if not nodes:
    print("No nodes found")
    exit(0)

node = nodes[0]
print(f"=== Node {node.id} Start Task Logs ===\n")

# Get stdout
try:
    stdout = batch_client.file.get_from_compute_node(pool_id, node.id, "startup/stdout.txt")
    print("--- STDOUT ---")
    for chunk in stdout:
        print(chunk.decode('utf-8'), end='')
    print()
except Exception as e:
    print(f"Could not get stdout: {e}")

# Get stderr
try:
    stderr = batch_client.file.get_from_compute_node(pool_id, node.id, "startup/stderr.txt")
    print("\n--- STDERR ---")
    for chunk in stderr:
        print(chunk.decode('utf-8'), end='')
    print()
except Exception as e:
    print(f"Could not get stderr: {e}")
