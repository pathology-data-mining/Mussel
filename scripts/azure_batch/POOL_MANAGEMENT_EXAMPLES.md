# Azure Batch Pool Management Examples

This document demonstrates the enhanced pool management capabilities in the Azure Batch submission scripts.

## Creating a Pool with GPU Support

By default, pools are created with GPU support enabled:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <key> \
  --batch-account-url https://mybatchaccount.batch.azure.com \
  --pool-id gpu-pool \
  --create-pool \
  --vm-size Standard_NC6s_v3 \
  --node-count 2 \
  --use-gpu \
  --container-image mskmind/mussel:latest-torch-gpu \
  --job-id myjob \
  --create-job
```

## Creating an Auto-Scaling Pool

Enable auto-scaling to dynamically adjust pool size based on workload:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <key> \
  --batch-account-url https://mybatchaccount.batch.azure.com \
  --pool-id autoscale-pool \
  --create-pool \
  --vm-size Standard_NC6s_v3 \
  --node-count 1 \
  --enable-auto-scale \
  --min-node-count 1 \
  --max-node-count 10 \
  --use-gpu \
  --container-image mskmind/mussel:latest-torch-gpu \
  --job-id myjob \
  --create-job
```

**How auto-scaling works:**
- Pool starts with minimum nodes (1 in this example)
- Scales up to maximum nodes (10 in this example) based on pending tasks
- Automatically scales down when tasks complete
- Evaluates workload every 15 minutes by default

## Creating a CPU-Only Pool

For CPU workloads, you can explicitly disable GPU support:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <key> \
  --batch-account-url https://mybatchaccount.batch.azure.com \
  --pool-id cpu-pool \
  --create-pool \
  --vm-size Standard_D4s_v3 \
  --node-count 4 \
  --no-gpu \
  --container-image mskmind/mussel:latest-torch-cpu \
  --job-id myjob \
  --create-job
```

## Automatic Pool Deletion After Task Completion

When you want to automatically clean up resources after processing completes, use the cleanup flags with `--monitor`:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name mybatchaccount \
  --batch-account-key <key> \
  --batch-account-url https://mybatchaccount.batch.azure.com \
  --pool-id temp-pool \
  --create-pool \
  --vm-size Standard_NC6s_v3 \
  --node-count 2 \
  --job-id temp-job \
  --create-job \
  --csv-manifest slides.csv \
  --output-s3-prefix s3://bucket/results/ \
  --aws-access-key-id $AWS_ACCESS_KEY_ID \
  --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --monitor \
  --delete-job \
  --delete-pool
```

**How it works:**
1. Pool is created
2. Job is created
3. Tasks are submitted
4. Script monitors progress until all tasks complete
5. Job is deleted
6. Pool is deleted

**⚠️ Important:** Always use `--delete-job` and `--delete-pool` together with `--monitor`. Using them without monitoring will delete resources immediately, potentially terminating running tasks.

## Complete Workflow Example

Here's a complete example that creates a pool, processes slides, and cleans up automatically:

```bash
#!/bin/bash

# Configuration
BATCH_ACCOUNT_NAME="mybatchaccount"
BATCH_ACCOUNT_KEY="<your-batch-key>"
BATCH_ACCOUNT_URL="https://mybatchaccount.eastus.batch.azure.com"
POOL_ID="mussel-pool-$(date +%Y%m%d-%H%M%S)"
JOB_ID="mussel-job-$(date +%Y%m%d-%H%M%S)"

# Process slides with automatic cleanup
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name "$BATCH_ACCOUNT_NAME" \
  --batch-account-key "$BATCH_ACCOUNT_KEY" \
  --batch-account-url "$BATCH_ACCOUNT_URL" \
  --pool-id "$POOL_ID" \
  --create-pool \
  --vm-size Standard_NC6s_v3 \
  --node-count 2 \
  --use-gpu \
  --container-image mskmind/mussel:latest-torch-gpu \
  --job-id "$JOB_ID" \
  --create-job \
  --csv-manifest manifest.csv \
  --output-s3-prefix s3://my-bucket/results/ \
  --aws-access-key-id "$AWS_ACCESS_KEY_ID" \
  --aws-secret-access-key "$AWS_SECRET_ACCESS_KEY" \
  --aws-region us-east-1 \
  --monitor \
  --delete-job \
  --delete-pool

echo "Processing complete and resources cleaned up!"
```

## Complete Workflow with Auto-Scaling

Here's an example using auto-scaling for large batch processing:

```bash
#!/bin/bash

# Configuration
BATCH_ACCOUNT_NAME="mybatchaccount"
BATCH_ACCOUNT_KEY="<your-batch-key>"
BATCH_ACCOUNT_URL="https://mybatchaccount.eastus.batch.azure.com"
POOL_ID="mussel-autoscale-pool-$(date +%Y%m%d-%H%M%S)"
JOB_ID="mussel-job-$(date +%Y%m%d-%H%M%S)"

# Process many slides with auto-scaling pool
python scripts/azure_batch/submit_batch_jobs.py \
  --batch-account-name "$BATCH_ACCOUNT_NAME" \
  --batch-account-key "$BATCH_ACCOUNT_KEY" \
  --batch-account-url "$BATCH_ACCOUNT_URL" \
  --pool-id "$POOL_ID" \
  --create-pool \
  --vm-size Standard_NC6s_v3 \
  --node-count 2 \
  --enable-auto-scale \
  --min-node-count 2 \
  --max-node-count 20 \
  --use-gpu \
  --container-image mskmind/mussel:latest-torch-gpu \
  --job-id "$JOB_ID" \
  --create-job \
  --csv-manifest large_manifest.csv \
  --output-s3-prefix s3://my-bucket/results/ \
  --aws-access-key-id "$AWS_ACCESS_KEY_ID" \
  --aws-secret-access-key "$AWS_SECRET_ACCESS_KEY" \
  --aws-region us-east-1 \
  --monitor \
  --delete-job \
  --delete-pool

echo "Processing complete and resources cleaned up!"
```

**Benefits of auto-scaling:**
- Start with minimum nodes (2 in this example) to save costs
- Automatically scale up to 20 nodes when there are many pending tasks
- Scale down when tasks complete to reduce costs
- No manual intervention needed

## Pool Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--pool-id` | Unique identifier for the pool | Required |
| `--create-pool` | Create the pool if it doesn't exist | Flag |
| `--vm-size` | Azure VM size | Standard_NC6s_v3 |
| `--node-count` | Number of VMs in the pool (or initial/min for auto-scale) | 1 |
| `--use-gpu` | Enable GPU support | True |
| `--no-gpu` | Disable GPU support | Flag |
| `--enable-auto-scale` | Enable auto-scaling | Flag |
| `--min-node-count` | Minimum nodes for auto-scaling | node-count |
| `--max-node-count` | Maximum nodes for auto-scaling | Required if auto-scale |
| `--auto-scale-evaluation-interval` | Evaluation interval in minutes | 15 |
| `--container-image` | Docker image to use | mskmind/mussel:latest-torch-gpu |

## Recommended VM Sizes

### GPU VMs (for faster processing)
- `Standard_NC6s_v3`: 6 vCPUs, 112 GB RAM, 1x NVIDIA V100 GPU
- `Standard_NC12s_v3`: 12 vCPUs, 224 GB RAM, 2x NVIDIA V100 GPU
- `Standard_NC24s_v3`: 24 vCPUs, 448 GB RAM, 4x NVIDIA V100 GPU

### CPU VMs (for cost-effective processing)
- `Standard_D4s_v3`: 4 vCPUs, 16 GB RAM
- `Standard_D8s_v3`: 8 vCPUs, 32 GB RAM
- `Standard_D16s_v3`: 16 vCPUs, 64 GB RAM

## Best Practices

1. **Use unique pool IDs**: Include timestamps to avoid conflicts
   ```bash
   POOL_ID="mussel-pool-$(date +%Y%m%d-%H%M%S)"
   ```

2. **Always monitor when auto-deleting**: Don't use `--delete-pool` without `--monitor`
   ```bash
   # Good: Delete after completion
   --monitor --delete-pool
   
   # Bad: Deletes immediately
   --delete-pool
   ```

3. **Match VM size to workload**:
   - GPU workloads: Use NC-series VMs with `--use-gpu`
   - CPU workloads: Use D-series VMs with `--no-gpu`

4. **Match container image to VM type**:
   - GPU VMs: Use `mskmind/mussel:latest-torch-gpu`
   - CPU VMs: Use `mskmind/mussel:latest-torch-cpu`

5. **Clean up resources**: Always use `--delete-job` and `--delete-pool` with `--monitor` to avoid unnecessary costs
