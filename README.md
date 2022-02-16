# Learning Shape Representations for Person Re-Identification under Clothing Change
This repository contains codes for WACV2021 CASE-Net

[[Paper]](https://openaccess.thecvf.com/content/WACV2021/papers/Li_Learning_Shape_Representations_for_Person_Re-Identification_Under_Clothing_Change_WACV_2021_paper.pdf)
Pytorch implementation for our WACV 2021 paper.

## Prerequisites
- Python 3
- [Pytorch](https://pytorch.org/)

## Getting Started

### Setup Dataset Root Directory
``` 
python setup.py --dataset-dir path/to/dataset
```

It will automatically generate a ```config.py``` under the current directory.

### Script for train
``` 
sh train-main.sh
```

### Setup for test (under code cleaning and construction)
``` 
sh test-main.sh
```

## Acknowledgements
Our code is HEAVILY modified from [DG-Net](https://github.com/NVlabs/DG-Net).

## Citation
Please cite our paper if you find the code useful for your research.
```
@inproceedings{li2021learning,
  title={Learning shape representations for person re-identification under clothing change},
  author={Li, Yu-Jhe and Weng, Xinshuo and Kitani, Kris M},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2432--2441},
  year={2021}
}
```