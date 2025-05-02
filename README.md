# Mussel

<img src="docs/mussel.jpg" width="300px" />

This is a fork of Faisal Mahmood's CLAM repository (GPL v3 license), with the following modifications:
- Added CTransPath and Quilt embeddings
- Added zero-shot tissue-type annotation of tiles
- Added browser for text-to-image AI search of MSK pathology data
- Added caching of images for inference right on the tiles (rather than on embeddings)
- Added microns per pixel (mpp) as parameter for tiling, supported regardless of native slide resolution
- Made usable for job submission (one script run, one slide)
- Removed modeling

## Installation

### System requirements

Supported systems:
* Mac OS (x86 and ARM)
* Linux (x86)

### Pre-requisites
- [uv](https://docs.astral.sh/uv/)
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### Create virtual environment and install packages

Model inference either uses the pytorch libraries or tensorflow. Unfortunately,
there are often conflicts with CUDA libraries required by both when running on
GPU. To handle this, the required set of libraries must be specified to `uv` to
build the virtual environment correctly.

#### PyTorch

Required for the following models:

* [ResNet-50](https://huggingface.co/microsoft/resnet-50)
* [TransPath](https://github.com/Xiyue-Wang/TransPath)
* [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath)
* [Virchow](https://huggingface.co/paige-ai/Virchow)
* [H-Optimus-0](https://huggingface.co/bioptimus/H-optimus-0)
* [OpenCLIP](https://github.com/mlfoundations/open_clip)


##### GPU (CUDA)

```bash
uv sync --extra torch-gpu
```

##### CPU

```bash
uv sync --extra torch-cpu
```

#### Tensorflow

Required for:

* [Google Path Foundation](https://huggingface.co/google/path-foundation)

##### GPU (CUDA)

```bash
uv sync --extra tensorflow-gpu
```

##### CPU

```bash
uv sync --extra tensorflow-cpu
```

## CLI

There are 4 CLI tools: `annotate`, `cache_tiles`, `extract_features`, and
`tessellate`. See CLI options using `{command} --help`.

### foreground detection and tiling

<img src="docs/example-mask.jpg" width="600px" />

Generate .h5 file with coordinates and metadata necessary for downstream steps. Optionally generate stitch and mask.

Example command (see defaults with `tessellate --help`):
```bash
mkdir reef
uv run tessellate \
    slide_path=data/7789726.svs \
    output_h5_path=reef/7789726_coord.h5 \
    seg_config.use_otsu=true
```

### feature extraction
Generate .h5 file with features and .pt file with embeddings for each tile.

Example command (see defaults with `extract_features --help`):
```bash
uv run extract_features \
    slide_path=data/7789726.svs \
    patch_h5_path=reef/7789726_coord.h5 \
    output_h5_path=reef/7789726_feat.h5 \
    output_pt_path=reef/7789726_embed.pt
```

#### (beta) - Folder-based feature extraction
Generates .h5 file with features using pre-tiled images (as opposed to tiles that come
from `tessellate`)
```bash
uv run extract_features \
    slide_path=None \
    patch_h5_path=None \
    patch_path=[path to folder w/ tiles in image format (.tif, .png, .jpg, etc.)] \
    output_h5_path=[path to output h5 file] \
    output_pt_path=None
```

### annotate tiles with tissue types (QuiltNet only)
Current classes are:
```
["carcinoma in situ", "invasive carcinoma with lymphocytes", "tumor infiltrating lymphocytes", "lymphocytes", "carcinoma in situ with lymphocytes", "tumor-associated stroma with lymphocytes"]
```

Try your own classes! Any natural language works, and no training is required.
Generate interrogation reports to eval your prompt engineering by setting `iterrogate`.

Create class embeddings
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
    patch_h5_path=reef/7789726_coord.h5 output_pt_path=reef/7789726_cache.pt \
    'limit_to_class=["carcinoma in situ", "invasive carcinoma with lymphocytes"]' \
    output_indices_json_path=reef/7789726_output_indices.json
```

*This takes about ten seconds for an example slide.*


## Development Notes

### Modifying package requirements

Add abstract requirements to the the `pyproject.toml`. Use `uv sync` to build `uv.lock`:

```bash
uv sync
```

### Run unit tests

Ensure that the dev group is installed (installed by default).

```bash
uv run pytest tests
```

### Create conda environment

Create a conda environment, activate it, and install mussel with:

```bash
uv pip install -r pyproject.toml
```

Specify the extra packages as required for your conda environment.

## License
This code is made available under the GPLv3 License and is available for non-commercial academic purposes.
Forked from CLAM, © [Mahmood Lab](http://www.mahmoodlab.org).

## Reference

Please cite the original CLAM [paper](https://www.nature.com/articles/s41551-020-00682-w):

Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w
```
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature Biomedical Engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021},
  publisher={Nature Publishing Group}
}
```
