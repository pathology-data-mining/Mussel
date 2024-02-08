# Mussel

## setup
```bash
conda create --name mussel -c conda-forge openslide-python numpy pandas opencv h5py matplotlib
conda activate mussel
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

Download modified timm from [here](https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view)
```bash
pip install timm-0.5.4.tar
```

## fast-patching
```bash
python create_patches_fp.py --patch --seg --stitch --save_dir {save-dir} --source {path-to-svs} --mpp 1.0 --step_size 224 --patch_size 224
```
Note: CTransPath strictly requires 224x224 patches

## feat extraction

### resnet50
```bash
python extract_features_fp.py --model resnet50 --save_dir {save-dir} --slide_file_path {path-to-svs} --patch_file-path {path-to-patch-file}
```



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
