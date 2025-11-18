# Azure Container Registry (ACR) Authentication Setup

## Problem

Azure Batch pool failed to pull Docker image with error:
```
Code: ContainerInvalidImage
Message: authentication required, visit https://aka.ms/acr/authorization
```

## Root Cause

Azure Batch doesn't have credentials to pull from your private Azure Container Registry (ACR).

**Current configuration** in `secrets.env`:
```bash
export AZURE_CONTAINER_REGISTRY_SERVER="mskocracontainerregister-cfbfchg8dgfbedan.azurecr.io"
# Missing: USERNAME and PASSWORD!
```

## Solution

Add ACR credentials to `secrets.env` file.

## Step 1: Get ACR Credentials

### Option A: Use Admin Credentials (Easiest)

```bash
# Enable admin user (if not already enabled)
az acr update --name mskocracontainerregister --admin-enabled true

# Get credentials
az acr credential show --name mskocracontainerregister

# Output:
# {
#   "passwords": [
#     {
#       "name": "password",
#       "value": "YOUR_PASSWORD_HERE"
#     },
#     {
#       "name": "password2",
#       "value": "YOUR_PASSWORD2_HERE"
#     }
#   ],
#   "username": "mskocracontainerregister"
# }
```

### Option B: Use Service Principal (More Secure)

```bash
# Create service principal with AcrPull role
az ad sp create-for-rbac \
    --name acr-batch-pull \
    --role acrpull \
    --scopes /subscriptions/<subscription-id>/resourceGroups/<rg>/providers/Microsoft.ContainerRegistry/registries/mskocracontainerregister

# Output:
# {
#   "appId": "YOUR_APP_ID",
#   "password": "YOUR_PASSWORD",
#   "tenant": "YOUR_TENANT"
# }

# Use appId as username, password as password
```

## Step 2: Add to secrets.env

Edit `secrets.env` and add:

```bash
# Azure Container Registry credentials
export AZURE_CONTAINER_REGISTRY_SERVER="mskocracontainerregister-cfbfchg8dgfbedan.azurecr.io"
export AZURE_CONTAINER_REGISTRY_USERNAME="mskocracontainerregister"  # or service principal appId
export AZURE_CONTAINER_REGISTRY_PASSWORD="YOUR_PASSWORD_HERE"
```

## Step 3: Verify Setup

```bash
# Source the updated secrets
source secrets.env

# Test that variables are set
echo $AZURE_CONTAINER_REGISTRY_SERVER
echo $AZURE_CONTAINER_REGISTRY_USERNAME
echo $AZURE_CONTAINER_REGISTRY_PASSWORD  # Should show password

# Test login
docker login $AZURE_CONTAINER_REGISTRY_SERVER \
    -u $AZURE_CONTAINER_REGISTRY_USERNAME \
    -p $AZURE_CONTAINER_REGISTRY_PASSWORD

# Should show: Login Succeeded
```

## Step 4: Run Azure Batch

Now when you submit jobs, the credentials will be automatically loaded:

```bash
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    --staging-container mussel-staging \
    --config config.yaml \
    --csv-manifest slides.csv
```

The script will:
1. Load ACR credentials from secrets.env
2. Configure pool with registry authentication
3. Azure Batch can now pull your private images!

## How It Works

### Environment Variables
```bash
AZURE_CONTAINER_REGISTRY_SERVER     # ACR hostname
AZURE_CONTAINER_REGISTRY_USERNAME   # Username or app ID
AZURE_CONTAINER_REGISTRY_PASSWORD   # Password or service principal password
```

### Code Flow
```python
# submit_batch_jobs.py loads from env
args.container_registry_server = os.environ.get("AZURE_CONTAINER_REGISTRY_SERVER")
args.container_registry_username = os.environ.get("AZURE_CONTAINER_REGISTRY_USERNAME")
args.container_registry_password = os.environ.get("AZURE_CONTAINER_REGISTRY_PASSWORD")

# Passed to pool creation
pool.container_configuration = ContainerConfiguration(
    container_image_names=[docker_image],
    container_registries=[
        ContainerRegistry(
            registry_server=container_registry_server,
            user_name=container_registry_username,
            password=container_registry_password,
        )
    ]
)
```

## Troubleshooting

### Error: "Login Succeeded" but Azure Batch still fails

**Possible causes**:
1. Credentials not in secrets.env (check with `grep CONTAINER_REGISTRY secrets.env`)
2. Not using `--env-file` flag when running script
3. Typo in variable names (must match exactly)

**Solution**:
```bash
# Verify env vars are loaded
python scripts/azure_batch/submit_batch_jobs.py \
    --env-file secrets.env \
    ... \
    2>&1 | grep "Container registry:"

# Should show:
#   Container registry: mskocracontainerregister-cfbfchg8dgfbedan.azurecr.io
```

### Error: "Access denied" or "insufficient permissions"

**Possible causes**:
- Service principal doesn't have `AcrPull` role
- Admin user disabled on ACR

**Solution**:
```bash
# Enable admin user
az acr update --name mskocracontainerregister --admin-enabled true

# Or grant service principal access
az role assignment create \
    --assignee <service-principal-app-id> \
    --role AcrPull \
    --scope /subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.ContainerRegistry/registries/mskocracontainerregister
```

### Error: "Image not found"

**Possible causes**:
- Image not pushed to ACR
- Wrong image name/tag

**Solution**:
```bash
# List images in ACR
az acr repository list --name mskocracontainerregister

# List tags for specific image
az acr repository show-tags --name mskocracontainerregister --repository mussel

# Should show: gigapath-latest
```

## Security Best Practices

### Don't Commit Secrets!
```bash
# Make sure secrets.env is in .gitignore
grep "secrets.env" .gitignore

# If not:
echo "secrets.env" >> .gitignore
```

### Use Service Principal (Production)
- ✅ Better security (limited scope)
- ✅ Can be rotated independently
- ✅ Auditable access
- ❌ More setup required

### Use Admin Credentials (Development)
- ✅ Easy setup
- ✅ Works immediately
- ❌ Full registry access
- ❌ Shared credentials

## Quick Setup Script

```bash
#!/bin/bash
# setup_acr_credentials.sh

# Get ACR name (without full URL)
ACR_NAME="mskocracontainerregister"

# Enable admin and get credentials
echo "Enabling admin user and fetching credentials..."
az acr update --name $ACR_NAME --admin-enabled true

# Get credentials
CREDENTIALS=$(az acr credential show --name $ACR_NAME)
USERNAME=$(echo $CREDENTIALS | jq -r '.username')
PASSWORD=$(echo $CREDENTIALS | jq -r '.passwords[0].value')

# Add to secrets.env
echo "" >> secrets.env
echo "# Azure Container Registry credentials" >> secrets.env
echo "export AZURE_CONTAINER_REGISTRY_SERVER=\"${ACR_NAME}-cfbfchg8dgfbedan.azurecr.io\"" >> secrets.env
echo "export AZURE_CONTAINER_REGISTRY_USERNAME=\"${USERNAME}\"" >> secrets.env
echo "export AZURE_CONTAINER_REGISTRY_PASSWORD=\"${PASSWORD}\"" >> secrets.env

echo "✓ Credentials added to secrets.env"
echo "Run: source secrets.env"
```

## Summary

**To fix the authentication error**:

1. Get ACR credentials:
   ```bash
   az acr credential show --name mskocracontainerregister
   ```

2. Add to `secrets.env`:
   ```bash
   export AZURE_CONTAINER_REGISTRY_USERNAME="mskocracontainerregister"
   export AZURE_CONTAINER_REGISTRY_PASSWORD="your_password_here"
   ```

3. Run with `--env-file`:
   ```bash
   python scripts/azure_batch/submit_batch_jobs.py --env-file secrets.env ...
   ```

**That's it!** Azure Batch will now be able to pull your private Docker images.
