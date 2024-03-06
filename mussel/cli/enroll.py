import pandas as pd
import os


def enroll_in_reef(csv_path, reef_dir="/gpfs/mskmind_ess/pdm/reef"):
    existing_inventory = pd.read_csv(f"{reef_dir}/slide_directory.csv")
    existing_inventory['image_id'] = existing_inventory['image_id'].astype(str)
    
    new_inventory = pd.read_csv(csv_path)
    new_inventory['image_id'] = new_inventory['image_id'].astype(str)

    new_slide_ids = set(new_inventory['image_id'])
    existing_slide_ids = set(existing_inventory['image_id'])
    common_slide_ids = new_slide_ids.intersection(existing_slide_ids)
    if len(common_slide_ids) > 0:
        raise ValueError(f"Slide IDs {common_slide_ids} already exist in the reef")
    
    assert len(new_inventory.columns) == 2, f"Expected 2 columns, got {len(new_inventory.columns)}"
    assert 'image_id' in new_inventory.columns, f"Expected 'image_id' in columns, got {new_inventory.columns}"
    assert 'slide_file' in new_inventory.columns, f"Expected 'slide_file' in columns, got {new_inventory.columns}"

    for img_path in new_inventory['slide_file']:
        assert os.path.exists(img_path), f"Path {img_path} does not exist"

    inventory = pd.concat([existing_inventory, new_inventory], ignore_index=True)
    inventory.to_csv(f"{reef_dir}/slide_directory.csv", index=False)
    print(f"Enrolled {len(new_inventory)} slides in the reef")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Enroll slides in the reef')
    parser.add_argument('csv_path', type=str, help='Path to csv with slide inventory')
    parser.add_argument('--reef_dir', type=str, default="/gpfs/mskmind_ess/pdm/reef", help='Path to reef directory')
    args = parser.parse_args()
    enroll_in_reef(args.csv_path, args.reef_dir)