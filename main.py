import argparse
import tessellate #, cache_tiles, extract_features, annotate
import pandas as pd
import os


def get_paths(args):
    paths = {}
    
    slide_inventory = pd.read_csv(os.path.join(args.reef_dir, 'slide_directory.csv'))
    paths['slide_path'] = slide_inventory[slide_inventory['image_id'] == args.image_id]['slide_file'].values[0]

    sub_reef_dir = os.path.join(args.reef_dir, f"{args.mpp}_{args.patch_size}_{args.step_size}_None")
    assert os.path.exists(sub_reef_dir), f"sub_reef_dir {sub_reef_dir} does not exist"

    paths['patch_path'] = os.path.join(paths['sub_reef_dir'], 'patches', f"{args.image_id}.h5")
    paths['stitch_path'] = os.path.join(paths['sub_reef_dir'], 'stitches', f"{args.image_id}.jpg")
    paths['mask_path'] = os.path.join(paths['sub_reef_dir'], 'masks', f"{args.image_id}.jpg")

    paths['cache_path'] = os.path.join(paths['sub_reef_dir'], 'cache', f"{args.image_id}.pt")
    paths['annot_path'] = os.path.join(paths['sub_reef_dir'], 'annot', f"{args.image_id}.csv")
    paths['annot_class_report_path'] = os.path.join(paths['sub_reef_dir'], 'annot_class_reports', f"{args.image_id}.csv")

    if 'model' in args:
        paths['pt_feats_path'] = os.path.join(paths['sub_reef_dir'], {args.model}, 'pt', f"{args.image_id}.pt")
        paths['h5_feats_path'] = os.path.join(paths['sub_reef_dir'], {args.model}, 'h5', f"{args.image_id}.h5")
    
    return paths


# Create the parser
parser = argparse.ArgumentParser(description='select a command')

# Add the arguments
parser.add_argument('command', type=str, help='tessellate, featurize, annotate, or cache_tiles')
parser.add_argument('image_id', type=str, help='image id')
parser.add_argument('--reef_dir', type=str, help='location of mussel reef', default='/gpfs/mskmind_ess/pdm/reef')
parser.add_argument('--mpp', type=float, help='microns per pixel', default=1.0)
parser.add_argument('--patch_size', type=int, help='patch size', default=224)
parser.add_argument('--step_size', type=int, help='step size', default=896)

# Add subparsers
subparsers = parser.add_subparsers()
parser_tessellate = subparsers.add_parser('tessellate')
args = parser.parse_args()

if args.command == 'tessellate':
    tessellate.main(args)


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

paths = get_paths(args)

# Run the function associated with the chosen subparser
args.func(args)