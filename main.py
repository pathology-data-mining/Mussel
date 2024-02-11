import argparse
import tessellate, extract_features #, cache_tiles, , annotate
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
    paths['annot_class_report_path'] = os.path.join(sub_reef_dir, 'annot_class_reports', f"{args.image_id}.csv")

    if 'model' in args:
        paths['pt_feats_path'] = os.path.join(sub_reef_dir, {args.model}, 'pt', f"{args.image_id}.pt")
        paths['h5_feats_path'] = os.path.join(sub_reef_dir, {args.model}, 'h5', f"{args.image_id}.h5")
    
    return paths


# Create the parser
parser = argparse.ArgumentParser(description='select a command')

# Add the arguments
parser.add_argument("command", help="subcommand to run")
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

args = parser.parse_args()
print(args)
paths = get_paths(args)

# run command
if args.command == 'tessellate':
    tessellate.main(in_path_wsi=paths['slide_path'],
                    out_path_patch=paths['patch_path'],
                    out_path_mask=paths['mask_path'],
                    out_path_stitch=paths['stitch_path'],
                    patch_size=args.patch_size,
                    step_size=args.step_size,
                    mpp=args.mpp)

elif args.command == 'featurize':
    extract_features.main(
        h5_feats_path=paths['h5_feats_path'],
        pt_feats_path=paths['pt_feats_path'],
        patch_path=paths['patch_path'],
        slide_path=paths['slide_path'],
        model_name=args.model_name,
        batch_size=args.batch_size,
        gpus=args.gpus,
)


# Create a parser for the 'tessellate' command
# Add arguments to the 'tessellate' parser here
# For example: parser_tessellate.add_argument('--option', help='Option for tessellate')
parser_tessellate.set_defaults(func=tessellate.main)

# Create a parser for the 'cache' command
parser_cache = subparsers.add_parser('featurize')
# Add arguments to the 'cache' parser here
# For example: parser_cache.add_argument('--option', help='Option for cache')
parser.add_argument('--model', type=str, help='model', default='quilt')
parser_cache.set_defaults(func=cache.main)

# Parse the arguments
args = parser.parse_args()


# Run the function associated with the chosen subparser
args.func(args)