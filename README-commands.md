# Mussel commands

This document provides a more detailed reference to the command-line tools provided by
Mussel.

Required for the following models:

* [ResNet-50](https://huggingface.co/microsoft/resnet-50)
* [TransPath](https://github.com/Xiyue-Wang/TransPath)
* [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath)
* [Virchow](https://huggingface.co/paige-ai/Virchow)
* [H-Optimus-0](https://huggingface.co/bioptimus/H-optimus-0)
* [OpenCLIP](https://github.com/mlfoundations/open_clip)


## Commands

Mussel provides handful of CLI tools:

* `tessellate` - tiling and foreground detection
* `extract_features` - generate embeddings with a pathology foundation model 
* `filter_features` - 
* `cache_tiles`
* `stitch_tiles`
* `annotate`
* `create_class_embeddings`
* `merge_annotation_features`
* `linear_probe_benchmark`
* `export_tiles`


Each of these commands is configurable with a number of different parameters, which are summarized
further below.  You can always get a quick list of the parameters and default values for a given tool
by executing `<command> --help`.

### Examples

<img src="docs/example-mask.jpg" width="600px" />

The example commands below use the test data provided in the `tests/testdata` folder. 

Some of the commands expect to write their outputs to a `reef/` folder in the current directory,
so you should create that directory before running them, with

    mkdir -p reef


### `tessellate`

Tessellate tiles a whole-slide image.  The tile coordinates and other metadata necessary
for downstream steps are written to an .h5 file.

Example command (see defaults with `tessellate --help`):
```bash
    tessellate \
        slide_path=tests/testdata/948176.svs \
        output_h5_path=reef/948176_coord.h5 \
        seg_config.segment_threshold=0 \
        num_workers=1
```

### feature extraction
Generate .h5 file with features and .pt file with embeddings for each tile.

Example command (see defaults with `extract_features --help`):
```bash
extract_features \
    slide_path=data/7789726.svs \
    patch_h5_path=reef/7789726_coord.h5 \
    output_h5_path=reef/7789726_feat.h5 \
    output_pt_path=reef/7789726_embed.pt
```

Generate a .h5 file with features calculated from a folder of pre-tiled images
(as opposed to the tiles that come from `tessellate`)
```bash
extract_features \
    slide_path=None \
    patch_h5_path=None \
    patch_path=<path to folder w/ tiles in image format (.tif, .png, .jpg, etc.)> \
    output_h5_path=<path to output h5 file> \
    output_pt_path=None
```

### annotate tiles with tissue types (QuiltNet only)
Current classes are:
```
["carcinoma in situ", "invasive carcinoma with lymphocytes", "tumor infiltrating lymphocytes", "lymphocytes", "carcinoma in situ with lymphocytes", "tumor-associated stroma with lymphocytes"]
```

Try your own classes! Any natural language works, and no training is required.
Generate interrogation reports to eval your prompt engineering by setting `iterrogate`.

### generate class embeddings
```bash
uv run create_class_embeddings \
    'classes=["carcinoma in situ", "invasive carcinoma with lymphocytes", "tumor infiltrating lymphocytes", "lymphocytes", "carcinoma in situ with lymphocytes", "tumor-associated stroma with lymphocytes" ]' \
    output_pt_path=reef/classes.pt
```

Example command (see defaults with `annotate --help`):
```bash
uv run annotate \
    features_pt_path=reef/7789726_embed.pt \
    output_csv_path=reef/7789726.csv \
    'classes=["carcinoma in situ", "invasive carcinoma with lymphocytes", "tumor infiltrating lymphocytes", "lymphocytes", "carcinoma in situ with lymphocytes", "tumor-associated stroma with lymphocytes" ]' \
    class_embedding_pt_path=reef/classes.pt
```

<img src="docs/example-interrog.png" width="600px" />

### tile caching

Generate .pt file for rapid access of tiles during I/O intense operations such
as training. This can be conditioned on tissue types: e.g. cache only the tiles
containing invasive carcinoma by setting `limit_to_class`. `patches_h5_path` is
the output from `tessellate`.

```bash
uv run cache_tiles slide_path=data/7789726.svs \
    patch_h5_path=reef/7789726_coord.h5 \
    output_pt_path=reef/7789726_cache.pt \
    'limit_to_class=["carcinoma in situ", "invasive carcinoma with lymphocytes"]' \
    output_indices_json_path=reef/7789726_output_indices.json
```

*This takes about ten seconds for an example slide.*




