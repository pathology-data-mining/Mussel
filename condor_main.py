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
        status = check_reef_status(image_id)
        if status[field_to_check]:
            print(f"{image_id} has already undergone {all_args['command']} in reef")
        else:
            image_ids_to_process.append(image_id)
    all_args['image_id'] = image_ids_to_process
    if len(all_args['image_id']) == 0:
        print(f"No images to process for {all_args['command']}")
        exit()
    return all_args

def submit_condor_jobs(mussel_params, n_jobs):
    job = htcondor.Submit({"executable": "data/patch_and_cache.sh",
                           "arguments": "$(ProcId) {mpp} {patch_size} {step_size} {norm}".format(**mussel_params),
                           "universe": "vanilla",
                           "request_gpus": 0,
                           "request_memory": "10GB",
                           "request_cpus": 8,
                           "output": "outputs/preprocessing/condor_logs/output.out",
                           "log": "outputs/preprocessing/condor_logs/log.log",
                           "error": "outputs/preprocessing/condor_logs/err.err"})
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(job, count=n_jobs)
    cluster_id = submit_result.cluster()
    print(f"Submitted HTCondor cluster with ID: {cluster_id}")
    return cluster_id    


if __name__ == "__main__":
    all_args = parse_args()
    assert len(all_args['image_id']) > 1, "use main.py for single image_id"
    all_args = filter_to_pending(all_args)
    for image_id in all_args['image_id']:
        args = all_args.copy()
        args['image_id'] = image_id
        main(args)

    # cluster_id = submit_condor_jobs(MUSSEL_PARAMS, len(SLIDES_TO_CACHE))
    
    # # wait until done
    # schedd = htcondor.Schedd()
    # cluster_ads = schedd.query(f"ClusterId == {cluster_id}")
    # while len(cluster_ads) > 0:
    #     print(f"Waiting for {len(cluster_ads)} jobs to finish...")
    #     time.sleep(30)
    #     cluster_ads = schedd.query(f"ClusterId == {cluster_id}")
