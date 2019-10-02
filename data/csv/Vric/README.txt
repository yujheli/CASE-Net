# Vehicle Re-Identificaition in Context

Vehicle Re-Identificaition in Contexta (VRIC) is a multi resolution vehicle re-identification dataset.
The images in the dataset are collcted from UA-Detrac detection and tracking becnhmark.
This dataset is research only and the license belongs to the original authors of the UA-Detrac benchmark.


******************************************************************************************************************
The VRIC dataset is collection of images extracted UA-Detrac detection and tracking benchmark.
The resulting dataset has:
2811 train identities with 54808 images
2811 test identities with 5622 images.


Annotations can be found in "vric_train.txt", "vric_probe.txt", "vric_gallery.txt".
The format of the files are the following each row correspoing to an image in the dataset.
[Image_name] [ID label] [Cam Label]

If you use this dataset, please kindly cite our paper as,
Kanaci, A., Zhu, X., Gong, S.: Vehicle Re-Identificaition in Context
German Conference on Patttern Recognition (2018)

@inproceedings{2018gcpr-Kanaci,
author    = {Aytac Kanaci and
             Xiatian Zhu and
             Shaogang Gong},
title     = {Vehicle Re-Identification in Context},
booktitle = {Pattern Recognition - 40th German Conference, {GCPR} 2018, Stuttgart,
             Germany, September 10-12, 2018, Proceedings},
year      = {2018}
}

and UA-Detrac
@article{DETRAC:CoRR:WenDCLCQLYL15,
  author    = {Longyin Wen and Dawei Du and Zhaowei Cai and Zhen Lei and Ming{-}Ching Chang and
               Honggang Qi and Jongwoo Lim and Ming{-}Hsuan Yang and Siwei Lyu},
  title     = { {DETRAC:} {A} New Benchmark and Protocol for Multi-Object Detection and Tracking},
  journal   = {arXiv CoRR},
  volume    = {abs/1511.04136},
  year      = {2015}
}

This dataset should be used for research only. Please DO NOT distribute or use it for commercial purpose.


For more information visit: qmul-vric.gtihub.io
******************************************************************************************************************

Content in the directory:
1. "probe_images/". This dir contains 2811 images as queries.
2. "gallery_images/". This dir contains 2811 images for testing.
3. "train_images/". This dir contains 54808 images for training.
4. "vric_probe.txt". It lists all query file names.
5. "vric_gallery.txt". It lists all test file names.
6. "vric_train.txt". It lists all train file names.
7. "Readme.txt". This readme file.
