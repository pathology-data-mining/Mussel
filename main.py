import argparse
import pandas as pd
import os
from reef_helpers import get_paths


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='select a command')

    # Add shared arguments
    parser.add_argument("command", help="tessellate, featurize, annotate, cache")
    parser.add_argument('--image_id', nargs='+', type=str, help='image id')
    parser.add_argument('--reef_dir', type=str, help='location of mussel reef', default='/gpfs/mskmind_ess/pdm/reef')
    parser.add_argument('--mpp', type=float, help='microns per pixel', default=1.0)
    parser.add_argument('--patch_size', type=int, help='patch size', default=224)
    parser.add_argument('--step_size', type=int, help='step size', default=896)

    # Add featurize arguments
    featurize_group = parser.add_argument_group('featurize', 'featurize options')
    featurize_group.add_argument('--model_name', type=str, help='model', default='quilt')
    featurize_group.add_argument('--gpus', nargs="+", type=int, default=[0])
    featurize_group.add_argument('--batch_size', type=int, default=64)

    # Add annotate arguments
    annotate_group = parser.add_argument_group('annotate', 'annotate options')
    annotate_group.add_argument('--interrogate', action='store_true', help='interrogate')

    # Add cache arguments
    cache_group = parser.add_argument_group('cache', 'cache options')
    cache_group.add_argument("--limit_to_class", type=str, default=None, help="limit to class")

    args = parser.parse_args()
    args = vars(args)
    return args


def main(args):
    if isinstance(args['image_id'], list):
        assert len(args['image_id']) == 1, "only one image_id is supported for main.py; use condor_main.py for multiple"
        args['image_id'] = str(args['image_id'][0])
    paths = get_paths(args)

    # run command
    if args['command'] == 'tessellate':
        import tessellate
        assert os.path.exists(paths['slide_path']), f"slide_path {paths['slide_path']} does not exist"
        tessellate.main(in_path_wsi=paths['slide_path'],
                        out_path_patch=paths['patch_path'],
                        out_path_mask=paths['mask_path'],
                        out_path_stitch=paths['stitch_path'],
                        patch_size=args['patch_size'],
                        step_size=args['step_size'],
                        mpp=args['mpp'])

    elif args['command'] == 'featurize':
        import extract_features
        assert os.path.exists(paths['patch_path']), f"patch_path {paths['patch_path']} does not exist"
        extract_features.main(
                h5_feats_path=paths['h5_feats_path'],
                pt_feats_path=paths['pt_feats_path'],
                patch_path=paths['patch_path'],
                slide_path=paths['slide_path'],
                model_name=args['model_name'],
                batch_size=args['batch_size'],
                gpus=args['gpus'])

    elif args['command'] == 'annotate':
        import annotate
        assert os.path.exists(paths['pt_feats_path']), f"pt_feats_path {paths['pt_feats_path']} does not exist"
        assert os.path.exists(paths['class_json_path']), f"class_json_path {paths['class_json_path']} does not exist"
        annotate.main(
            slide_emb_path=paths['pt_feats_path'],
            class_json_path=paths['class_json_path'],
            output_path=paths['annot_path'],
            interrogate=args['interrogate'],
            svs_path=paths['slide_path'],
            patch_path=paths['patch_path'],
            interrogation_report_path=paths['annot_class_report_path'])

    elif args['command'] == 'cache':
        import cache_tiles
        assert os.path.exists(paths['slide_path']), f"slide_path {paths['slide_path']} does not exist"
        assert os.path.exists(paths['patch_path']), f"patch_path {paths['patch_path']} does not exist"
        if args['limit_to_class']:
            assert os.path.exists(paths['annot_path']), f"annot_path {paths['annot_path']} does not exist"
        cache_tiles.main(
            slide_file_path=paths['slide_path'],
            patch_file_path=paths['patch_path'],
            output_path=paths['cache_path'],
            limit_to_class=args['limit_to_class'],
            annot_path=paths['annot_path'],
            cache_tile_indices_path=paths['cache_tile_indices_path'])


if __name__ == "__main__":
    args = parse_args()
    main(args)
