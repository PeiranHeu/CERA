![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)  

# CERA
> Pytorch implementation for **Cross-Space Multimodal Emotion Recognition with Heterogeneous Feature Association**

## Overview
<p align="center">
<img src='img\\architecture.PNG'/>


CERA comprises three modules: Heterogeneous Feature Decoupling (HFD), Distribution-based Subspace Association (DSA) and Cross-Space Feature Fusion (CFF). HFD extracts heterogeneous features from multimodal data, and separates them into modality-invariant and modality-specific subspaces. DSA and CFF implement space association, where DSA associates intra-subspace or inter-subspace complementary features, and CFF fuses the two groups of features for prediction.

## Usage


### Datasets

[CMU-MOSI CMU-MOSEI](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)

### get started

1. Set up the environment (need conda prerequisite)

```
conda create -n env_name python==3.7
bash init.sh
```

2. Modify the data path in train.sh and start training

```
bash train.sh
```


