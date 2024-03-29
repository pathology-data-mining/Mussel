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

Usable using Condor, but suffers from limitations of current Condor system:
- sporadic data accessibility (e.g. sparky2 cannot access `/gpfs/mskmind_emc/data_large`)
- uncontrolled GPU usage by non-Condor usage (e.g. user was using most gpus across pll1,2,3 without Condor when I tried to test the featurization using Condor, so mine largely failed due to clashes.)

## Installation

### Pre-requisites

-   [mamba](https://mamba.readthedocs.io/en/latest/installation.html):
    Follow the instructions for installing
    [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge)

### Create virtual environment and install packages

```bash
mamba env create -p .venv/ -f environment.yaml
mamba activate .venv/
poetry install
```

### If you plan to use CTransPath

Install CTransPath somewhere and reference the path in the config yaml as
`transpath_dir`. Reference to the CTransPath model (.pth file) with
`transpath_model_path`.

## CLI

There are 4 CLI tools: `annotate`, `cache_tiles`, `extract_features`, and
`tessellate`. See CLI options using `{command} --help`.

### foreground detection and tiling

<img src="docs/example-mask.jpg" width="600px" />

Generate `.h5` file with coordinates and metadata. Optionally generate stitch and mask.

Example command (see defaults with `tessellate --help`):
```bash
tessellate slide_path=[slide path] output_h5_path=[output path] seg_config.use_otsu=true
```

### feature extraction
Generate .h5 file and .pt file with embeddings for each tile.

Example command (see defaults with `extract_features --help`):
```bash
extract_features slide_path=[slide path] patch_h5_path=[patch path] output_h5_path=[output h5 path] output_pt_path=[output pt path]
```

### annotate tiles with tissue types (QuiltNet only)
Current classes are:
```
{"0": "benign epithelium", "1": "carcinoma in situ", "2": "invasive carcinoma", "3": "connective tissue", "4": "adipose", "5": "vessel", "6": "necrosis", "7": "marking pen"}
```

Try your own classes! Any natural language works, and no training is required.
Generate interrogation reports to eval your prompt engineering by setting `iterrogate`.

Example command (see defaults with `annotate --help`):
```bash
annotate features_pt_path=[features pt path] output_csv_path=[output csv path] class_json_path=[custom class json file]
```

<img src="docs/example-interrog.png" width="600px" />

### tile caching

Generate .pt file for rapid access of tiles during I/O intense operations such
as training. This can be conditioned on tissue types: e.g. cache only the tiles
containing invasive carcinoma by setting `limit_to_class`. `patches_h5_path` is
the output from `tessellate`.

```bash
cache_tiles slide_path=[slide path] patches_h5_path=[patches h5 path] output_pt_path=[output pt path] 'limit_to_class=[adipose,invasive carcinoma]' output_indices_json_path=[output indices json]
```

*This takes about ten seconds for an example slide.*

## SnakeMake Pipeline

Run the snakemake pipeline on a set of slides to build a shareable 'reef' of
slide patches and extracted features.

### Configuration

Create a `config/config.yaml` file that conforms to the specification in
`config/config.schema.yaml`. `slide_list_path` must be set to a file containing
the list of image IDs that you wish to analyze. The image IDs must exist in the
`slide_directory_csv_path`, which is a csv file that must conform to
`config/slides.schema.yaml`. If you set a `parameters_path` to override
defaults, make sure that conforms to the `config/params.schema.yaml` in .csv
format (i.e. columns are parameters and rows are the parameter values).

### Using HT-Condor for job submission

Install the [HT-Condor profile](https://github.com/msk-mind/snakemake-htcondor)
using [cookiecutter](https://github.com/cookiecutter/cookiecutter). Note that
the condor job directory must be in a network-shared location. After the
profile is installed, you can use it with:

```bash
snakemake --profile {name of htcondor profile}
```

### foreground detection and tiling

Generate `.h5` files with coordinates and metadata in the `reef_dir`.

```bash
snakemake --cores {number of jobs} tiles
```
*This takes about one second for an example slide. Using Condor, tiling 1000 slides takes about two minutes.*

### feature extraction
Generate .h5 and .pt files with embeddings for each tile in the `reef_dir`. (This is the default snakemake target.)

```bash
snakemake --cores {number of jobs}
```

*This takes about 30 seconds for an example slide.*

### annotate tiles with tissue types (QuiltNet only)

Set `annotation_class_json_path` to a custom json file. Annotations will be created in a subdirectory of `output_dir`.

```bash
snakemake --cores {number of cores} annotations
```

### tile caching

Generate .pt file for rapid access of tiles during I/O intense operations such
as training. This can be conditioned on tissue types: e.g. cache only the tiles
containing invasive carcinoma by setting `cache_limit_to_class` in the `config/config.yaml` file.

```bash
snakemake --cores {number of cores} cached_tiles
```

### output files

Slide tiling and extracted features are output to the `reef_dir`:

```
reef
└── model--quiltnet
    └── mpp--0.5
        └── patch_size--256
            └── step_size--256
                └── tissue_area_threshold--100
                    ├── features
                    │   ├── h5
                    │   │   ├── 1051151.h5
                    │   │   ├── 1064170.h5
                    │   │   ├── 759112.h5
                    │   │   ├── 979373.h5
                    │   │   └── 980143.h5
                    │   └── pt
                    │       ├── 1051151.pt
                    │       ├── 1064170.pt
                    │       ├── 759112.pt
                    │       ├── 979373.pt
                    │       └── 980143.pt
                    ├── patches
                    │   ├── 1051151.h5
                    │   ├── 1064170.h5
                    │   ├── 759112.h5
                    │   ├── 979373.h5
                    │   └── 980143.h5
                    └── stitches
                        ├── 1051151.jpg
                        ├── 1064170.jpg
                        ├── 759112.jpg
                        ├── 979373.jpg
                        └── 980143.jpg
```

Annotations and cached tiles output to the `output_dir`:

```
results
└── model--quiltnet
    └── mpp--0.5
        └── patch_size--256
            └── step_size--256
                └── tissue_area_threshold--100
                    ├── annotations
                    │   ├── 1051151.csv
                    │   ├── 1064170.csv
                    │   ├── 759112.csv
                    │   ├── 979373.csv
                    │   └── 980143.csv
                    └── cache_tiles
                        ├── indices
                        │   ├── 1051151.json
                        │   ├── 1064170.json
                        │   ├── 759112.json
                        │   ├── 979373.json
                        │   └── 980143.json
                        └── pt
                            ├── 1051151.pt
                            ├── 1064170.pt
                            ├── 759112.pt
                            ├── 979373.pt
                            └── 980143.pt
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
