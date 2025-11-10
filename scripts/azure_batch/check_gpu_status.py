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

# Check pool configuration
pool_id = "mussel-pool"
pool = batch_client.pool.get(pool_id)

print(f"Pool: {pool_id}")
print(f"VM Size: {pool.vm_size}")
print(f"Current Dedicated Nodes: {pool.current_dedicated_nodes}")
print(f"Current Low Priority Nodes: {pool.current_low_priority_nodes}")
print(f"Target Dedicated Nodes: {pool.target_dedicated_nodes}")
print(f"Target Low Priority Nodes: {pool.target_low_priority_nodes}")
print(f"Allocation State: {pool.allocation_state}")

# Check nodes
print("\nNodes in pool:")
nodes = list(batch_client.compute_node.list(pool_id))
for node in nodes[:3]:
    print(f"  Node {node.id}: State={node.state}, VM Size={pool.vm_size}")

print(f"\nTotal nodes: {len(nodes)}")
