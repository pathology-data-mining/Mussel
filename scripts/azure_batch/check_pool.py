#!/usr/bin/env python3
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
import os
import yaml

with open('../../azure_test.yaml', 'r') as f:
    config = yaml.safe_load(f)

credentials = SharedKeyCredentials(
    config['azure']['batch_account_name'],
    os.environ['AZURE_BATCH_ACCOUNT_KEY']
)
batch_client = BatchServiceClient(
    credentials,
    batch_url=config['azure']['batch_account_url']
)

# Check pool
try:
    pool = batch_client.pool.get('mussel-pool')
    print(f'Pool: {pool.id}')
    print(f'  VM Size: {pool.vm_size}')
    print(f'  State: {pool.allocation_state}')
    print(f'  Current dedicated: {pool.current_dedicated_nodes}')
    print(f'  Target dedicated: {pool.target_dedicated_nodes}')
    
    # Check for resize errors
    if pool.resize_errors:
        print(f'\n  Resize errors:')
        for error in pool.resize_errors:
            print(f'    Code: {error.code}')
            print(f'    Message: {error.message}')
            if error.values:
                for detail in error.values:
                    print(f'      - {detail.name}: {detail.value}')
    
    # Check nodes
    nodes = list(batch_client.compute_node.list('mussel-pool'))
    print(f'\n  Nodes: {len(nodes)}')
    for node in nodes[:3]:
        print(f'    - {node.id}: {node.state}')
        if node.start_task_info:
            print(f'      Start task: {node.start_task_info.state}')
    
except Exception as e:
    print(f'Error: {e}')
