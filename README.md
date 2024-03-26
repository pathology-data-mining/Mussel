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

## setup
```bash
mamba env create -p .venv/ -f environment.yaml
mamba activate .venv/
poetry install
```

### Configuration

Create a `config/config.yaml` file that conforms to the specification in
`config/config.schema.yaml`. `slide_list_path` must be set to a file containing the
list of image IDs that you wish to analyze. The image IDs must exist in the
`slide_directory_path`. If you set a `parameters_path` to override defaults,
make sure that conforms to the `config/params.schema.yaml` in .csv format (i.e.
columns are parameters and rows are the parameter values).

### If you plan to use CTransPath

Install CTransPath somewhere and reference the path in the config yaml as
`transpath_dir`. Reference to the CTransPath model with `transpath_model_path`.

### Using HT-Condor for job submission

Install the [HT-Condor profile](https://github.com/msk-mind/snakemake-htcondor)
using [cookiecutter](https://github.com/cookiecutter/cookiecutter). After the
profile is installed, you can use it with:

```bash
snakemake --profile {name of htcondor profile}
```

## foreground detection and tiling
<img src="docs/example-mask.jpg" width="600px" />
Generate `.h5` file with coordinates and metadata necessary for downstream steps in the `reef_dir`.

```bash
snakemake --cores {number of jobs} tiles
```
*This takes about one second for an example slide. Using Condor, tiling 1000 slides takes about two minutes.*

## feat extraction
Generate .h5 file and .pt file with embeddings for each tile in the `reef_dir`. (This is the default snakemake target.)

```bash
snakemake --cores {number of jobs}
```

*This takes about 30 seconds for an example slide.*

## annotate tiles with tissue types (beta feature)
Current classes are:
```
{"0": "benign epithelium", "1": "carcinoma in situ", "2": "invasive carcinoma", "3": "connective tissue", "4": "adipose", "5": "vessel", "6": "necrosis", "7": "marking pen"}
```

Try your own classes! Any natural language works, and no training is required.
Generate interrogation reports to eval your prompt engineering. Set
`annotation_class_json_path` to a custom json file.

```bash
snakemake --cores {number of cores} annotations
```

<img src="docs/example-interrog.png" width="600px" />

## tile caching

Generate .pt file for rapid access of tiles during I/O intense operations such
as training. This can be conditioned on tissue types: e.g. cache only the tiles
containing invasive carcinoma by setting `cache_limit_to_cass` in the `config/config.yaml` file.

```bash
snakemake --cores {number of cores} cached_tiles
```

*This takes about ten seconds for an example slide.*

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
