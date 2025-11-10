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

pool = batch_client.pool.get("mussel-pool")

print(f"Pool: mussel-pool")
print(f"VM Size: {pool.vm_size}")
print(f"Image Reference:")
print(f"  Publisher: {pool.virtual_machine_configuration.image_reference.publisher}")
print(f"  Offer: {pool.virtual_machine_configuration.image_reference.offer}")
print(f"  SKU: {pool.virtual_machine_configuration.image_reference.sku}")
print(f"Node Agent SKU: {pool.virtual_machine_configuration.node_agent_sku_id}")
print(f"Container Config Type: {pool.virtual_machine_configuration.container_configuration.type}")
