#!/usr/bin/env python3
"""
Stage slides from S3 to Azure Files for batch processing.

This script stages slides from S3 (or local paths) to Azure Files,
which can then be used by Azure Batch tasks without re-uploading.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))

try:
    from azure_files_staging import AzureFilesStaging
except ImportError:
    print("ERROR: Could not import azure_files_staging module.")
    print("Make sure scripts/common/azure_files_staging.py is available.")
    sys.exit(1)


def stage_slides_from_csv(
    csv_path: str,
    account_name: str,
    account_key: str,
    share_name: str,
    remote_dir: str = "slides",
    output_csv: str = None,
    resume_from: int = 0,
    limit: int = None,
) -> dict:
    """
    Stage slides from CSV manifest to Azure Files.
    
    Args:
        csv_path: Path to CSV with columns: image_id, sample_id, svs_path
        account_name: Azure Storage account name
        account_key: Azure Storage account key
        share_name: Azure Files share name
        remote_dir: Remote directory within share (default: "slides")
        output_csv: Path to write CSV with azfiles:// paths
        resume_from: Skip first N slides (for resuming)
        limit: Maximum number of slides to stage (for testing)
    
    Returns:
        Dictionary mapping image_id to azfiles:// path
    """
    # Initialize Azure Files staging
    staging = AzureFilesStaging(
        account_name=account_name,
        account_key=account_key,
        share_name=share_name,
    )
    
    # Read CSV
    slides = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slides.append({
                'image_id': row['image_id'],
                'sample_id': row['sample_id'],
                'slide_path': row['svs_path'],
            })
    
    total_slides = len(slides)
    print(f"Found {total_slides} slides in manifest")
    
    # Apply resume and limit
    if resume_from > 0:
        print(f"Resuming from slide {resume_from + 1}")
        slides = slides[resume_from:]
    
    if limit:
        slides = slides[:limit]
        print(f"Limiting to {limit} slides")
    
    print(f"Staging {len(slides)} slides to Azure Files...")
    print(f"  Share: {share_name}")
    print(f"  Remote directory: {remote_dir}")
    print("")
    
    # Stage each slide
    staged_paths = {}
    failed = []
    
    for idx, slide in enumerate(slides, 1):
        image_id = slide['image_id']
        sample_id = slide['sample_id']
        slide_path = slide['slide_path']
        
        # Determine filename from path
        filename = os.path.basename(slide_path)
        
        # Build remote path
        remote_path = f"{remote_dir}/{filename}"
        
        # Build azfiles:// URL
        azfiles_url = f"azfiles://{account_name}/{share_name}/{remote_path}"
        
        try:
            print(f"[{idx + resume_from}/{total_slides}] Staging {image_id} ({sample_id})")
            print(f"  From: {slide_path}")
            print(f"  To:   {azfiles_url}")
            
            # Upload (handles S3 download if needed)
            staging.upload_file(
                local_path=slide_path,
                remote_path=remote_path,
                show_progress=False,
            )
            
            staged_paths[image_id] = azfiles_url
            print(f"  ✓ Success")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed.append({
                'image_id': image_id,
                'sample_id': sample_id,
                'slide_path': slide_path,
                'error': str(e),
            })
        
        print("")
    
    # Write output CSV if requested
    if output_csv:
        print(f"Writing staged paths to: {output_csv}")
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['slide_id', 'slide_path'])
            writer.writeheader()
            for image_id, azfiles_path in staged_paths.items():
                # Find sample_id for this image_id (use as slide_id)
                sample_id = next(
                    (s['sample_id'] for s in slides if s['image_id'] == image_id),
                    image_id  # Fallback to image_id if sample_id not found
                )
                writer.writerow({
                    'slide_id': sample_id,
                    'slide_path': azfiles_path,
                })
        print(f"✓ Wrote {len(staged_paths)} paths to {output_csv}")
    
    # Report summary
    print("")
    print("=" * 60)
    print("STAGING SUMMARY")
    print("=" * 60)
    print(f"Total slides in manifest: {total_slides}")
    print(f"Attempted to stage: {len(slides)}")
    print(f"Successfully staged: {len(staged_paths)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("")
        print("Failed slides:")
        for fail in failed:
            print(f"  {fail['image_id']} ({fail['sample_id']}): {fail['error']}")
    
    return staged_paths


def main():
    parser = argparse.ArgumentParser(
        description="Stage slides from S3 to Azure Files for batch processing"
    )
    parser.add_argument(
        "--csv-manifest",
        required=True,
        help="CSV file with columns: image_id, sample_id, svs_path",
    )
    parser.add_argument(
        "--storage-account-name",
        required=True,
        help="Azure Storage account name",
    )
    parser.add_argument(
        "--storage-account-key",
        required=True,
        help="Azure Storage account key",
    )
    parser.add_argument(
        "--azure-files-share-name",
        required=True,
        help="Azure Files share name",
    )
    parser.add_argument(
        "--remote-dir",
        default="slides",
        help="Remote directory within Azure Files share (default: slides)",
    )
    parser.add_argument(
        "--output-csv",
        help="Write CSV with azfiles:// paths for staged slides",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume from slide N (skip first N slides, for resuming failed runs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to staging first N slides (for testing)",
    )
    parser.add_argument(
        "--aws-access-key-id",
        help="AWS access key ID (for downloading from S3)",
    )
    parser.add_argument(
        "--aws-secret-access-key",
        help="AWS secret access key (for downloading from S3)",
    )
    parser.add_argument(
        "--aws-endpoint-url",
        help="AWS S3 endpoint URL (for custom S3 endpoints)",
    )
    
    args = parser.parse_args()
    
    # Set AWS credentials if provided
    if args.aws_access_key_id:
        os.environ['AWS_ACCESS_KEY_ID'] = args.aws_access_key_id
    if args.aws_secret_access_key:
        os.environ['AWS_SECRET_ACCESS_KEY'] = args.aws_secret_access_key
    if args.aws_endpoint_url:
        os.environ['AWS_ENDPOINT_URL'] = args.aws_endpoint_url
    
    # Stage slides
    staged_paths = stage_slides_from_csv(
        csv_path=args.csv_manifest,
        account_name=args.storage_account_name,
        account_key=args.storage_account_key,
        share_name=args.azure_files_share_name,
        remote_dir=args.remote_dir,
        output_csv=args.output_csv,
        resume_from=args.resume_from,
        limit=args.limit,
    )
    
    if len(staged_paths) == 0:
        print("\n⚠️  WARNING: No slides were successfully staged!")
        sys.exit(1)
    
    print("\n✓ Staging completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
