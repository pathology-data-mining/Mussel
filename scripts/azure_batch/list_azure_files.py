#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '../common')

from azure.storage.fileshare import ShareServiceClient

# Get credentials
storage_account = "mskpdmgen2"
share_name = "mussel-staging"
output_dir = "outputs"

storage_key = os.environ.get("AZURE_STORAGE_KEY")
if not storage_key:
    print("ERROR: AZURE_STORAGE_KEY not set")
    sys.exit(1)

account_url = f"https://{storage_account}.file.core.windows.net"
service_client = ShareServiceClient(account_url=account_url, credential=storage_key)

share_client = service_client.get_share_client(share_name)
directory_client = share_client.get_directory_client(output_dir)

print(f"Listing files in {storage_account}/{share_name}/{output_dir}:")
print("=" * 80)

try:
    files = list(directory_client.list_directories_and_files())
    if not files:
        print("No files found!")
    else:
        for item in files:
            if hasattr(item, 'size'):
                size_mb = item.size / (1024*1024)
                print(f"  {item.name} ({size_mb:.2f} MB)")
            else:
                print(f"  {item.name}/ (directory)")
        print(f"\nTotal items: {len(files)}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
