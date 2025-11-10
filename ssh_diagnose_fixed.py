#!/usr/bin/env python3
import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials

# Get credentials
batch_account_name = "ocra"
batch_account_key = os.environ['AZURE_BATCH_ACCOUNT_KEY']
batch_account_url = "https://ocra.eastus2.batch.azure.com"

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_account_url)

# Get pool information
pool_id = 'mussel-pool'
try:
    pool = batch_client.pool.get(pool_id)
    print(f"Pool: {pool_id}")
except Exception as e:
    print(f"Error getting pool: {e}")
    exit(1)

# List nodes
nodes = list(batch_client.compute_node.list(pool_id))
print(f"Found {len(nodes)} nodes")

# Check each node
for node in nodes:
    print(f"\n{'='*60}")
    print(f"Node: {node.id}")
    print(f"State: {node.state}")
    print(f"IP Address: {node.ip_address if hasattr(node, 'ip_address') else 'N/A'}")
    
    # Get remote login settings
    try:
        remote_login_settings = batch_client.compute_node.get_remote_login_settings(pool_id, node.id)
        print(f"\nSSH Connection Info:")
        print(f"  IP: {remote_login_settings.remote_login_ip_address}")
        print(f"  Port: {remote_login_settings.remote_login_port}")
        
        print(f"\nTo SSH manually:")
        print(f"  ssh -p {remote_login_settings.remote_login_port} pooladmin@{remote_login_settings.remote_login_ip_address}")
    except Exception as e:
        print(f"Could not get remote login settings: {e}")
    
    # Check file mounts
    print(f"\nFile Mounts:")
    if node.start_task_info:
        print(f"  Start task state: {node.start_task_info.state}")
        if node.start_task_info.failure_info:
            print(f"  Start task failure: {node.start_task_info.failure_info}")

# Get task details
job_id = 'mussel-test-job'
try:
    tasks = list(batch_client.task.list(job_id))
    print(f"\n{'='*60}")
    print(f"=== Tasks in job {job_id} ===")
    for task in tasks:
        print(f"\nTask {task.id}:")
        print(f"  State: {task.state}")
        print(f"  Node: {task.node_info.node_id if task.node_info else 'N/A'}")
        if task.execution_info:
            print(f"  Exit code: {task.execution_info.exit_code}")
            if task.execution_info.failure_info:
                print(f"  Failure: {task.execution_info.failure_info.category} - {task.execution_info.failure_info.message}")
except Exception as e:
    print(f"Error getting tasks: {e}")
