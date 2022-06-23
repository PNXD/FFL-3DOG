# FFL-3DOG
Free-form Description-guided 3D Visual Graph Networks for Object Grounding in Point Cloud

We visualize how the 3D grounding performs after VoteNet, nodes pruning and final result.
![image1](https://github.com/PNXD/FFL-3DOG/blob/main/visual.png)

## Setup
The code is now compatiable with PyTorch 1.6.

```conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch```

Then please run the following command to compile the CUDA module for the PointNet++ backbone.

```cd lib/pointnet2```

```python setup.py install```

## Data preparation
1. Download the preprocessed GLoVE embeddings and put them under data/.  
2. Download the ScanRefer dataset and unzip it under data/.  
3. Download the ScanNetV2 dataset and put ```scans/``` under ```data/scannet/scans/```.  
4. Running the following command to preprocess ScanNet data.  

```cd data/scannet/```  

```python batch_load_scannet_data.py```

## Training
You can train our model by running the scripts  

```python scripts/train.py --use_color --use_normal --use_pretrained```  


Thanks to [ScanRefer](https://github.com/daveredrum/ScanRefer).
