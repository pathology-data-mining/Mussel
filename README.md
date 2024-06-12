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

## Using the installed Mussel on the MSK-MIND servers

Mussel is installed in `/gpfs/mskmind_ess/mussel`. If you have `conda` already
installed, you can activate an existing Mussel virtual environment.

```bash
conda activate /gpfs/mskmind_ess/mussel/venv
```

If `conda` is not installed, either install it or first run:
```bash
 . "/gpfs/mskmind_ess/limr/mambaforge/etc/profile.d/conda.sh"
```

With the virtual environment activated, you can now run all the CLI 
commands.

## Installation

### Pre-requisites

- [mamba](https://mamba.readthedocs.io/en/latest/installation.html):
    Follow the instructions for installing
    [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge)
- [conda-lock](https://conda.github.io/conda-lock/)

### Create virtual environment and install packages

```bash
conda-lock install -p .venv/
mamba activate .venv/
pip install --no-deps .
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
python -m mussel.cli.tessellate \
    slide_path=data/7789726.svs \
    output_h5_path=reef/7789726_cood.h5 \
    seg_config.use_otsu=true
```

### feature extraction
Generate .h5 file with features and .pt file with embeddings for each tile.

Example command (see defaults with `extract_features --help`):
```bash
python -m mussel.cli.extract_features \
    slide_path=data/7789726.svs \
    patch_h5_path=reef/7789726_cood.h5 \
    output_h5_path=reef/7789726_feat.h5 \
    output_pt_path=reef/7789726_embed.pt
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
python -m mussel.cli.create_class_embeddings \
    'classes=["carcinoma in situ", "invasive carcinoma with lymphocytes", "tumor infiltrating lymphocytes", "lymphocytes", "carcinoma in situ with lymphocytes", "tumor-associated stroma with lymphocytes" ]' \
    output_pt_path=reef/classes.pt
```

Example command (see defaults with `annotate --help`):
```bash
python -m mussel.cli.annotate \
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
python -m mussel.cli.cache_tiles slide_path=data/7789726.svs \
    patch_h5_path=reef/7789726_cood.h5 output_pt_path=reef/7789726_cache.pt \
    'limit_to_class=["carcinoma in situ", "invasive carcinoma with lymphocytes"]' \
    output_indices_json_path=reef/7789726_output_indices.json
```

*This takes about ten seconds for an example slide.*


## Development Notes

### Modifying package requirements

Add abstract requirements to the conda `environment.yaml`, and build a new `conda-lock.yml` file:

```bash
conda-lock -f environment.yaml --with-cuda 12.3 # change cuda version as desired
```

Add the concrete requirements to the `requirements.txt` or `requirements-dev.txt`:

```bash
pip list --format=freeze > requirements.txt
```

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
