from snakemake.utils import Paramspace
import pandas as pd
import numpy as np
from dacite import from_dict
import dacite
from omegaconf import OmegaConf

from snakemake.utils import validate

import mussel.cli.tessellate
import mussel.cli.extract_features
from mussel.cli.tessellate import TessellateConfig, SegConfig, PatchConfig, FilterConfig, VisConfig
from mussel.cli.extract_features import Model, ExtractFeaturesConfig

configfile: "config.yaml"
validate(config, "config.schema.yaml")

slide_df = pd.read_csv(config['slide_directory_file'])
if config['test']:
  slide_df = slide_df[0:5]
validate(slide_df, "slides.schema.yaml")
slide_df.set_index("image_id", inplace=True, drop=False)

parameters = dict()
if config.get('parameters_file'):
  parameters = pd.read_csv(config['parameters_file'])
validate(parameters, "params.schema.yaml")

 # sort the parameters for consistent paramspace patterns
if type(parameters) == dict:
  df = pd.DataFrame([parameters])
  df = df.reindex(sorted(df.columns), axis=1)
  paramspace = Paramspace(df)
else:
  parameters = parameters.reindex(sorted(parameters.columns), axis=1)
  paramspace = Paramspace(parameters)

def get_slide(wildcards):
  return slide_df.loc[wildcards.slide_id, "slide_file"]

rule all:
  input:
    expand("reef/{params}/features/pt/{slide_id}.pt", params=paramspace.instance_patterns, slide_id=slide_df.index)

rule tessellate:
  input:
    get_slide
  output:
    patch_path=f"reef/{paramspace.wildcard_pattern}/patches/{{slide_id}}.h5",
    stitch_path=f"reef/{paramspace.wildcard_pattern}/stitches/{{slide_id}}.jpg"
  resources:
    mem_mb=30000
  run:
    # convert np types to native python types
    instance = { k:val.item() if type(val).__module__ == np.__name__ else val for k, val in paramspace.instance(wildcards).items() }
    cfg = TessellateConfig(slide_path=input[0],
                           output_path=output.patch_path,
                           stitch_path=output.stitch_path,
                           seg_config=from_dict(data_class=SegConfig, data=instance),
                           patch_config=from_dict(data_class=PatchConfig, data=instance),
                           filter_config=from_dict(data_class=FilterConfig, data=instance),
                           vis_config=from_dict(data_class=VisConfig, data=instance),
                           )
    mussel.cli.tessellate.main(OmegaConf.create(cfg))


rule featurize:
  input:
    patch_path=f"reef/{paramspace.wildcard_pattern}/patches/{{slide_id}}.h5",
    slide_path=get_slide
  output:
    pt=f"reef/{paramspace.wildcard_pattern}/features/pt/{{slide_id}}.pt",
    h5=f"reef/{paramspace.wildcard_pattern}/features/h5/{{slide_id}}.h5",
  threads: 32
  resources:
    mem_mb=30000,
    request_gpus=1
  run:
    instance = { k:val.item() if type(val).__module__ == np.__name__ else val for k, val in paramspace.instance(wildcards).items() }
    instance['patch_path'] = input.patch_path
    instance['output_h5_path'] = output.h5
    instance['output_pt_path'] = output.pt
    instance['slide_path'] = input.slide_path
    instance['num_workers'] = threads
    cfg = from_dict(data_class=ExtractFeaturesConfig, data=instance, config=dacite.Config(cast=[Model]))
    mussel.cli.extract_features.main(OmegaConf.create(cfg))

