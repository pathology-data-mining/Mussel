#!/usr/bin/env python3
import os
from azure.storage.fileshare import ShareFileClient

storage_account_name = "mskpdmgen2"
share_name = "mussel-staging"
storage_account_key = os.environ['AZURE_STORAGE_KEY']

# Upload classifier
classifier_path = "/gpfs/mskmind_ess/limr/repos/Mussel/model-1727990346535.pkl"
remote_path = "models/model-1727990346535.pkl"

print(f"Uploading {classifier_path} to {remote_path}...")

file_client = ShareFileClient(
    account_url=f"https://{storage_account_name}.file.core.windows.net",
    share_name=share_name,
    file_path=remote_path,
    credential=storage_account_key
)

with open(classifier_path, "rb") as f:
    file_client.upload_file(f)

print(f"Upload complete!")
