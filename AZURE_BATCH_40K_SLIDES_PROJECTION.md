# Azure Batch 40,000 Slides Processing Time Projection

## Test Data from Azure Batch (3 slides per task)

From the successful Azure Batch test run:

| Model | Time per 3 slides | Exit Code |
|-------|------------------|-----------|
| OPTIMUS | 6:26 (386s) | 0 |
| VIRCHOW2 | 3:53 (233s) | 0 |
| UNI2 | 1:13 (73s) | 0 |
| TITAN_SLIDE | 1:08 (68s) | 0 |
| GIGAPATH_SLIDE | 1:49 (109s) | 0 ‚úÖ (NOW FIXED) |

## Configuration

- **Total slides:** 40,000
- **Available GPUs:** 50 A100 GPUs
- **Node configuration:** 4 A100 GPUs per node = 12.5 nodes (round to 13 nodes)
- **Processing mode:** All 5 models run sequentially per slide
- **Slides per task:** 3 (optimal batch size from testing)
- **Tasks:** 40,000 / 3 = 13,334 tasks

## Time Calculations

### Time per 3 slides (all 5 models):
- OPTIMUS: 386s
- VIRCHOW2: 233s  
- UNI2: 73s
- TITAN_SLIDE: 68s
- GIGAPATH_SLIDE: 109s
- **Total per task: 869 seconds (14.5 minutes)**

### Parallel Execution with 50 GPUs:
- Total tasks: 13,334
- Tasks per GPU: 13,334 / 50 = 266.68 tasks per GPU
- Time per GPU: 266.68 √ó 869s = 231,745 seconds

### **Total Processing Time: 64.4 hours (2.7 days)**

## Breakdown by Model (40K slides)

| Model | Time/3 slides | Total GPU-hours | Wall time (50 GPUs) |
|-------|---------------|-----------------|---------------------|
| OPTIMUS | 6:26 | 1,432 hrs | 28.6 hours |
| VIRCHOW2 | 3:53 | 864 hrs | 17.3 hours |
| UNI2 | 1:13 | 271 hrs | 5.4 hours |
| TITAN_SLIDE | 1:08 | 252 hrs | 5.0 hours |
| GIGAPATH_SLIDE | 1:49 | 404 hrs | 8.1 hours |
| **TOTAL** | **14:29** | **3,223 hrs** | **64.4 hours** |

## Cost Estimation (Azure A100 80GB)

**Azure pricing (East US):**
- A100 80GB: ~$3.67/hour per GPU
- 50 GPUs √ó 64.4 hours = 3,220 GPU-hours
- **Total cost: ~$11,817**

## Optimizations to Consider

### 1. Increase GPU Count
- 100 GPUs: **32.2 hours** (~$11,817)
- 200 GPUs: **16.1 hours** (~$11,817)
- Cost stays same, but faster completion

### 2. Optimize Slide Batching
- Current: 3 slides per task
- Could try: 5-10 slides per task to reduce overhead
- Potential savings: 10-15% time reduction

### 3. Model Parallelization
- Run different models on different GPUs simultaneously
- Could reduce wall time significantly
- Requires different batch job structure

### 4. Use Spot Instances
- Azure Low Priority VMs: ~80% cost savings
- Risk: May be preempted
- Best for non-urgent workloads
- **Potential cost: ~$2,363**

## Recommended Approach

**Option 1: Standard (50 GPUs, On-Demand)**
- Time: 64.4 hours (2.7 days)
- Cost: ~$11,817
- Reliability: High
- Best for: Time-sensitive production runs

**Option 2: Cost-Optimized (100 GPUs, Low Priority)**
- Time: 32.2 hours (1.3 days)
- Cost: ~$2,363 (80% savings)
- Reliability: Medium (may need reruns)
- Best for: Research/development workloads

**Option 3: Balanced (75 GPUs, Low Priority with fallback)**
- Time: 43 hours (1.8 days)
- Cost: ~$3,150
- Reliability: Medium-High
- Best for: Most use cases

## Node Requirements for 50 GPUs

- Nodes needed: 50 GPUs / 4 GPUs per node = **13 nodes**
- Node SKU: Standard_ND96asr_v4 (4x A100 80GB)
- Total memory: 13 nodes √ó 900GB = 11.7TB RAM
- Total GPU memory: 50 GPUs √ó 80GB = 4TB GPU RAM

## Risk Mitigation

1. **Checkpointing:** Save progress every 1000 slides
2. **Retry logic:** Automatically retry failed tasks
3. **Monitoring:** Real-time progress dashboard
4. **Validation:** Spot-check outputs throughout run
5. **Backup plan:** Keep 20% spare GPU capacity for reruns

## Summary

búÖ **All 5 models now working** (including GIGAPATH_SLIDE)  
bè±Ô∏è **Estimated time: 64.4 hours** with 50 A100 GPUs  
püí∞ **Estimated cost: $11,817** (on-demand) or **$2,363** (low priority)  
püìä **Throughput: 621 slides/hour** or **14,904 slides/day**

