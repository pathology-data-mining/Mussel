from snakemake.utils import Paramspace
import pandas as pd
from dacite import from_dict
import OmegaConf

from snakemake.utils import validate

configfile: "config.yaml"
validate(config, "config.schema.yaml")

slide_df = pd.read_csv(config['slide_directory_file'])
validate(slide_df, "slides.schema.yaml")
slide_df.set_index("image_id", inplace=True, drop=False)

params_df = None
if config['parameters_file']:
  params_df = pd.read_csv(config['parameters_file'])
  validate(params_df, "params.schema.yaml")

default_params = config['parameters']

merged_params = []
if params_df:
  merged_params.append(default_params)
else:
  for _, row in params_df.iterrows():
    merged_params.append(dict(k, row.get(k, v) for k, v in default_params.items()))

paramspace = Paramspace(pd.DataFrame.from_records(merged_params))

rule all:
  output:
    expand(f"{paramspace.wildcard_pattern}/{slide_id}.pt", slide_id=slide_df.index)

rule tessellate:
  output:
    patch_path=f"{paramspace.wildcard_pattern}/patches/{slide_id}.h5",
    stitch_path=f"{paramspace.wildcard_pattern}/stitches/{slide_id}.jpg"
  params:
    slide_path = slide_df.loc[slide_id, "slide_file"]
  run:
    cfg = TessellateConfig(slide_path=params.slide_path,
                           output_path=output.patch_path,
                           stitch_path=output.stitch_path,
                           seg_config=from_dict(data_class=SegConfig, data=paramspace.instance),
                           patch_config=from_dict(data_class=PatchConfig, data=paramspace.instance),
                           filter_config=from_dict(data_class=FilterConfig, data=paramspace.instance),
                           vis_config=from_dict(data_class=VisConfig, data=paramspace.instance),
                           )
    seg_and_patch(cfg)


rule featurize:
  input:
    f"{paramspace.wildcard_pattern}/patches/{slide_id}.h5"
  output:
    f"{paramspace.wildcard_pattern}/{slide_id}.pt"
  params:
    slide_path = slide_df.loc[slide_id, "slide_file"]
  run:
    cfg = from_dict(data_class=ExtractFeaturesConfig, data=paramspace.instance)
    cfg.patch_path = input[0]
    cfg.output_path = output[0]
    cfg.slide_path = params.slide_path
    extract_features(cfg)

