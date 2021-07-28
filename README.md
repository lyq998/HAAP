# Homogeneous Architecture Augmentation for Neural Predictor

This repository is for the paper "Homogeneous Architecture Augmentation for Neural Predictor" which is accepted by ICCV 2021.

## Introduction

The codes have been tested on Python 3.6.

Dependent packages:

- nasbench (see  https://github.com/google-research/nasbench)
- nas_201_api (see https://github.com/D-X-Y/NAS-Bench-201)
- tensorflow (==1.15.0)
- scikit-learn
- matplotlib
- scipy

The *pkl* folder saves the fixed training set and fixed test set. 

If you would like to check other training data from NAS-Bench-101, please download the NAS-Bech-101 subset of the dataset with only models trained at 108 epochs: https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord. (More details are in https://github.com/google-research/nasbench, and you may be required to install additional dependencies like TensorFlow.) Then put the file *nasbench_only108.tfrecord* under the *path* folder. Finally, carefully delete the corresponding files in *pkl* folder.

## How to use

*Demo0.py* is for Table 1 and Figure 5. If you want to see *NPNAS + HA*, run the *neuralpredictor.pytorch/train.py* (--arch_aug is an argument). 

*GAon201.py* is for Table 3.

*Demo1.py* is for Table 4.

*GAon101/random_forest_Surrogate.py* is the Demo2, which is for Table 2. And this must need to download the file *nasbench_only108.tfrecord*. You can find the results in *GAon101/pops_log*.

*Demo3.py* is for Table 5.

*Demo5.py* is for Figure 7. If you want to test randomly, please delete the *pkl/num_creations.pkl*.

You can run these scripts to get the results reported in paper. You can change parameter settings following the annotations between codes.