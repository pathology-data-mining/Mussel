# Mussel

<img src="mussel.jpg" width="300px" />

This is a fork of Faisal Mahmood's CLAM repository (GPL v3 license), with the following modifications:
- Added microns per pixel (mpp) as parameter for tiling, supported regardless of native slide resolution
- Added CTransPath embeddings (in addition to ResNet50)
- Added caching of images for inference right on the tiles (rather than on embeddings)
- Made usable for job submission (one script run, one slide)
- Removed modeling

Missing feature: Macenko normalization

## setup
```bash
conda create --name mussel -c conda-forge \
openslide-python numpy pandas opencv h5py matplotlib

conda activate mussel
conda install pytorch torchvision pytorch-cuda=12.1 \
-c pytorch -c nvidia
```

If you plan to use CTransPath, download modified timm from [here](https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view)
```bash
pip install timm-0.5.4.tar
```

## tiling
Generate .h5 file with coordinates and metadata necessary for downstream steps, using the following arguments
- mpp (microns per pixel)
- step_size (distance between patches)
- patch_size (edge length of patch, must be 224 for CTransPath)
- save_dir (directory with or in which to create {masks,stitches,patches} subdirectories)

The patches directory contains the actual h5 file with coordinates and metadata required for downstream use. For quality control, masks show the tissue area, stitches show the tiling pattern.
```bash
python tessellate.py \
--save_dir {save-dir} \
--slide_file_path {path-to-svs} \
--mpp 1.0 \
--step_size 224 \
--patch_size 224
```
*This takes about one second for an example slide.*

## feat extraction
Generate .h5 file and .pt file with embeddings for each tile, using the following arguments
- model (resnet50 or ctranspath)
- save_dir (directory with or in which to create {h5_files,pt_files} subdirectories)

The h5_files directory contains the h5 file with embeddings (N_tiles x 768 or 1024), the pt_files directory contains the pt file with embeddings.

### resnet50
```bash
python extract_features.py \
--model resnet50 \
--save_dir {save-dir} \
--slide_file_path {path-to-svs} \
--patch_file-path {path-to-patch-file}
```

### ctranspath
```bash
python extract_features.py \
--model ctranspath \
--save_dir {save-dir} \
--slide_file_path {path-to-svs} \
--patch_file-path {path-to-patch-file}
```

*This takes about 30 seconds for an example slide.*

## tile caching
Generate .pt file for rapid access of tiles during I/O intense operations such as training.
```bash
python cache_tiles.py \
--slide_file_path {path-to-svs} \
--patch_file_path {path-to-patch-file} \
--output_path {path-to-pt-file}
```

*This takes about ten seconds for an example slide.*

## License
© [Mahmood Lab](http://www.mahmoodlab.org) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

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
