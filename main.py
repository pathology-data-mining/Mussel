import argparse
import pandas as pd
import os


def get_paths(args):
    paths = {}
    
    slide_inventory = pd.read_csv(os.path.join(args.reef_dir, 'slide_directory.csv'))
    slide_inventory['image_id'] = slide_inventory['image_id'].astype(str)
    paths['slide_path'] = slide_inventory[slide_inventory['image_id'] == str(args.image_id)]['slide_file'].values[0]

    sub_reef_dir = os.path.join(args.reef_dir, f"{args.mpp}_{args.patch_size}_{args.step_size}_None")
    assert os.path.exists(sub_reef_dir), f"sub_reef_dir {sub_reef_dir} does not exist"

    paths['patch_path'] = os.path.join(sub_reef_dir, 'patches', f"{args.image_id}.h5")
    paths['stitch_path'] = os.path.join(sub_reef_dir, 'stitches', f"{args.image_id}.jpg")
    paths['mask_path'] = os.path.join(sub_reef_dir, 'masks', f"{args.image_id}.jpg")

    paths['cache_path'] = os.path.join(sub_reef_dir, 'cache', f"{args.image_id}.pt")
    paths['annot_path'] = os.path.join(sub_reef_dir, 'annot', f"{args.image_id}.csv")
    paths['annot_class_report_path'] = os.path.join(sub_reef_dir, 'annot_class_reports', f"{args.image_id}.html")

    if args.command != 'tessellate':
        paths['pt_feats_path'] = os.path.join(sub_reef_dir, 'feats', args.model_name, 'pt', f"{args.image_id}.pt")
        paths['h5_feats_path'] = os.path.join(sub_reef_dir, 'feats', args.model_name, 'h5', f"{args.image_id}.h5")
        paths['class_json_path'] = os.path.join(sub_reef_dir, 'classes.json')
    
    return paths


# Create the parser
parser = argparse.ArgumentParser(description='select a command')

# Add the arguments
parser.add_argument("command", help="tesellate, featurize, annotate, cache")
parser.add_argument('image_id', type=str, help='image id')
parser.add_argument('--reef_dir', type=str, help='location of mussel reef', default='/gpfs/mskmind_ess/pdm/reef')
parser.add_argument('--mpp', type=float, help='microns per pixel', default=1.0)
parser.add_argument('--patch_size', type=int, help='patch size', default=224)
parser.add_argument('--step_size', type=int, help='step size', default=896)

tessellate_group = parser.add_argument_group('tessellate', 'tessellate options')

featurize_group = parser.add_argument_group('featurize', 'featurize options')
featurize_group.add_argument('--model_name', type=str, help='model', default='quilt')
featurize_group.add_argument('--gpus', nargs="+", type=int, default=[0])
featurize_group.add_argument('--batch_size', type=int, default=64)

annotate_group = parser.add_argument_group('annotate', 'annotate options')
annotate_group.add_argument('--interrogate', action='store_true', help='interrogate')

args = parser.parse_args()
paths = get_paths(args)

# run command
if args.command == 'tessellate':
    import tessellate
    assert os.path.exists(paths['slide_path']), f"slide_path {paths['slide_path']} does not exist"
    tessellate.main(in_path_wsi=paths['slide_path'],
                    out_path_patch=paths['patch_path'],
                    out_path_mask=paths['mask_path'],
                    out_path_stitch=paths['stitch_path'],
                    patch_size=args.patch_size,
                    step_size=args.step_size,
                    mpp=args.mpp)

elif args.command == 'featurize':
    import extract_features
    assert os.path.exists(paths['patch_path']), f"patch_path {paths['patch_path']} does not exist"
    extract_features.main(
            h5_feats_path=paths['h5_feats_path'],
            pt_feats_path=paths['pt_feats_path'],
            patch_path=paths['patch_path'],
            slide_path=paths['slide_path'],
            model_name=args.model_name,
            batch_size=args.batch_size,
            gpus=args.gpus)

elif args.command == 'annotate':
    import annotate
    assert os.path.exists(paths['pt_feats_path']), f"pt_feats_path {paths['pt_feats_path']} does not exist"
    assert os.path.exists(paths['class_json_path']), f"class_json_path {paths['class_json_path']} does not exist"
    annotate.main(
        slide_emb_path=paths['pt_feats_path'],
        class_json_path=paths['class_json_path'],
        output_path=paths['annot_path'],
        interrogate=args.interrogate,
        svs_path=paths['slide_path'],
        patch_path=paths['patch_path'],
        interrogation_report_path=paths['annot_class_report_path'])

elif args.command == 'cache':
    import cache_tiles
    assert os.path.exists(paths['slide_path']), f"slide_path {paths['slide_path']} does not exist"
    assert os.path.exists(paths['patch_path']), f"patch_path {paths['patch_path']} does not exist"
    cache_tiles.main(
        slide_file_path=paths['slide_path'],
        patch_file_path=paths['patch_path'],
        output_path=paths['cache_path']
    )
