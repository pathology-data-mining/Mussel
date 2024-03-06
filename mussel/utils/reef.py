import os
import pandas as pd


def get_paths(args):
    paths = {}
    
    slide_inventory = pd.read_csv(os.path.join(args['reef_dir'], 'slide_directory.csv'))
    slide_inventory['image_id'] = slide_inventory['image_id'].astype(str)
    try:
        paths['slide_path'] = slide_inventory[slide_inventory['image_id'] == str(args['image_id'])]['slide_file'].values[0]
    except IndexError:
        return None

    sub_reef_dir = os.path.join(args['reef_dir'], f"{args['mpp']}_{args['patch_size']}_{args['step_size']}_None")
    assert os.path.exists(sub_reef_dir), f"sub_reef_dir {sub_reef_dir} does not exist"

    paths['patch_path'] = os.path.join(sub_reef_dir, 'patches', f"{args['image_id']}.h5")
    paths['stitch_path'] = os.path.join(sub_reef_dir, 'stitches', f"{args['image_id']}.jpg")
    paths['mask_path'] = os.path.join(sub_reef_dir, 'masks', f"{args['image_id']}.jpg")

    paths['cache_path'] = os.path.join(sub_reef_dir, 'cache', f"{args['image_id']}.pt")
    paths['cache_tile_indices_path'] = os.path.join(sub_reef_dir, 'cache_tile_indices', f"{args['image_id']}.json")
    paths['annot_path'] = os.path.join(sub_reef_dir, 'annot', f"{args['image_id']}.csv")
    paths['annot_class_report_path'] = os.path.join(sub_reef_dir, 'annot_class_reports', f"{args['image_id']}.html")

    paths['pt_feats_path'] = os.path.join(sub_reef_dir, 'feats', args['model_name'], 'pt', f"{args['image_id']}.pt")
    paths['h5_feats_path'] = os.path.join(sub_reef_dir, 'feats', args['model_name'], 'h5', f"{args['image_id']}.h5")
    paths['class_json_path'] = os.path.join(sub_reef_dir, 'classes.json')
    paths['class_emb_path'] = os.path.join(sub_reef_dir, 'classes.pt')
    
    return paths


def check_reef_status(slide_id, model_name='quilt', mpp=1.0, patch_size=224, step_size=896, reef_dir="/gpfs/mskmind_ess/pdm/reef"):
    status = {}
    args = {'image_id': slide_id, 'model_name': model_name, 'mpp': mpp, 'patch_size': patch_size, 'step_size': step_size, 'reef_dir': reef_dir}
    paths = get_paths(args)
    if not paths:
        return None, None
    status['slide_exists'] = os.path.exists(paths['slide_path'])
    status['patch_exists'] = os.path.exists(paths['patch_path'])
    status['stitch_exists'] = os.path.exists(paths['stitch_path'])
    status['mask_exists'] = os.path.exists(paths['mask_path'])
    status['cache_exists'] = os.path.exists(paths['cache_path'])
    status['annot_exists'] = os.path.exists(paths['annot_path'])
    status['pt_feats_exists'] = os.path.exists(paths['pt_feats_path'])
    status['h5_feats_exists'] = os.path.exists(paths['h5_feats_path'])
    return status, paths
