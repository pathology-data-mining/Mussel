import htcondor
import time
from main import parse_args, main
from reef_helpers import check_reef_status

def filter_to_pending(all_args):
    if all_args['command'] == 'tessellate':
        field_to_check = 'patch_exists'
    elif all_args['command'] == 'featurize':
        field_to_check = 'pt_feats_exists'
    elif all_args['command'] == 'annotate':
        field_to_check = 'annot_exists'
    elif all_args['command'] == 'cache':
        field_to_check = 'cache_exists'

    # check status of each slide
    image_ids_to_process = []
    for image_id in all_args['image_id']:
        temp_args = all_args.copy()
        temp_args['image_id'] = image_id
        status = check_reef_status(temp_args)
        if status[field_to_check]:
            print(f"{image_id} has already undergone {all_args['command']} in reef")
        elif not status['slide_exists']:
            print(f"{image_id} source .svs file not found")
        else:
            image_ids_to_process.append(image_id)
    all_args['image_id'] = image_ids_to_process
    if len(all_args['image_id']) == 0:
        print(f"No images to process for {all_args['command']}")
        exit()
    return all_args, len(all_args['image_id'])

def format_args(all_args):
    arg_str = "$(ProcID) "
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

def submit_condor_jobs(all_args):
    all_args['image_ids_formatted'] = " ".join(all_args['image_id'])
    sub = htcondor.Submit({
        'executable': 'condor_mussel.sh',
        'universe': 'vanilla',
        'request_gpus': 1 if all_args['command'] == 'featurize' else 0,
        'request_memory': '30GB' if all_args['command'] == 'featurize' else '15GB',
        'request_cpus': 32 if all_args['command'] == 'featurize' else 16,
        'output': 'scratch/condor_output.out',
        'log': 'scratch/condor_log.log',
        'error': 'scratch/condor_err.err',
        "arguments": "$(Arguments)"
    })
    arguments = []
    for image_id in all_args['image_id']:
        temp_args = all_args.copy()
        temp_args['image_id'] = image_id
        print(temp_args['image_id'])
        assert isinstance(temp_args['image_id'], str)
        whether_to_interrogate = '--interrogate' if temp_args['interrogate'] else ''
        arguments.append({"Arguments": f"{temp_args['command']} --image_id {image_id} --reef_dir {temp_args['reef_dir']} --mpp {temp_args['mpp']} --patch_size {temp_args['patch_size']} --step_size {temp_args['step_size']} --model_name {temp_args['model_name']} --gpus {' '.join([str(x) for x in temp_args['gpus']])} --batch_size {temp_args['batch_size']} {whether_to_interrogate} --limit_to_class '{temp_args['limit_to_class']}'"})
    schedd = htcondor.Schedd()
    result = schedd.submit(sub, itemdata=iter(arguments))
    cluster_id = result.cluster()
    print(f"Submitted HTCondor cluster with ID: {cluster_id}")
    return cluster_id    


if __name__ == "__main__":
    ALL_ARGS = parse_args()
    print(ALL_ARGS)
    assert len(ALL_ARGS['image_id']) > 1, "use main.py for single image_id"
    ALL_ARGS, N_JOBS = filter_to_pending(ALL_ARGS)
    CLUSTER_ID = submit_condor_jobs(ALL_ARGS)
    
    # # wait until done
    SCHEDD = htcondor.Schedd()
    CLUSTER_ADS = SCHEDD.query(f"ClusterId == {CLUSTER_ID}")
    while len(CLUSTER_ADS) > 0:
        print(f"Waiting for {len(CLUSTER_ADS)} jobs to finish...")
        time.sleep(10)
        CLUSTER_ADS = SCHEDD.query(f"ClusterId == {CLUSTER_ID}")
