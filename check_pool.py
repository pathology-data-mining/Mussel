import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

batch_account_name = "ocra"
batch_account_key = os.environ['AZURE_BATCH_ACCOUNT_KEY']
batch_account_url = "https://ocra.eastus2.batch.azure.com"

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_account_url)

pool_id = "mussel-pool"

print(f"\n=== Pool {pool_id} Status ===\n")
try:
    pool = batch_client.pool.get(pool_id)
    print(f"State: {pool.state}")
    print(f"Allocation state: {pool.allocation_state}")
    print(f"VM size: {pool.vm_size}")
    print(f"Target dedicated nodes: {pool.target_dedicated_nodes}")
    print(f"Current dedicated nodes: {pool.current_dedicated_nodes}")
    print(f"Target low priority nodes: {pool.target_low_priority_nodes}")
    print(f"Current low priority nodes: {pool.current_low_priority_nodes}")
    
    if pool.resize_errors:
        print(f"\nResize errors:")
        for error in pool.resize_errors:
            print(f"  Code: {error.code}")
            print(f"  Message: {error.message}")
            if error.values:
                print(f"  Details:")
                for detail in error.values:
                    print(f"    - {detail.value} (Code: {detail.name})")
    
    # Check node status
    print(f"\n=== Nodes ===")
    nodes = list(batch_client.compute_node.list(pool_id))
    print(f"Total nodes: {len(nodes)}")
    for node in nodes:
        print(f"\nNode {node.id}:")
        print(f"  State: {node.state}")
        if node.start_task_info:
            print(f"  Start task state: {node.start_task_info.state}")
            if node.start_task_info.failure_info:
                print(f"  Start task failure: {node.start_task_info.failure_info.category} - {node.start_task_info.failure_info.message}")
                
except Exception as e:
    print(f"Error: {e}")
