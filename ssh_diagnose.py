#!/usr/bin/env python3
import os
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
import subprocess

# Get credentials from environment
batch_account_name = os.environ['AZURE_BATCH_ACCOUNT_NAME']
batch_account_key = os.environ['AZURE_BATCH_ACCOUNT_KEY']
batch_account_url = os.environ['AZURE_BATCH_ACCOUNT_URL']

credentials = SharedKeyCredentials(batch_account_name, batch_account_key)
batch_client = BatchServiceClient(credentials, batch_account_url)

# Get pool information
pool_id = 'mussel-pool'
pool = batch_client.pool.get(pool_id)
print(f"Pool: {pool_id}")

# List nodes
nodes = list(batch_client.compute_node.list(pool_id))
print(f"Found {len(nodes)} nodes")

# Get first node
if nodes:
    node = nodes[0]
    print(f"\nNode: {node.id}")
    print(f"State: {node.state}")
    
    # Get remote login settings
    remote_login_settings = batch_client.compute_node.get_remote_login_settings(pool_id, node.id)
    print(f"\nSSH Connection Info:")
    print(f"  IP: {remote_login_settings.remote_login_ip_address}")
    print(f"  Port: {remote_login_settings.remote_login_port}")
    
    # Create SSH command
    ssh_user = "pooladmin"
    ssh_host = remote_login_settings.remote_login_ip_address
    ssh_port = remote_login_settings.remote_login_port
    
    print(f"\nTo SSH manually:")
    print(f"  ssh -p {ssh_port} {ssh_user}@{ssh_host}")
    
    # Get task details to see what's happening
    job_id = 'mussel-test-job'
    tasks = list(batch_client.task.list(job_id))
    print(f"\n=== Running Tasks ===")
    for task in tasks:
        if task.state.value in ['running', 'active']:
            print(f"\nTask {task.id}:")
            print(f"  State: {task.state}")
            print(f"  Node: {task.node_info.node_id if task.node_info else 'N/A'}")
