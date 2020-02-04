# Single Camera Training for Person Re-identification

This repo includes the code for [Single Camera Training for Person Re-identification](https://arxiv.org/abs/1909.10848
). In utils/loss.py, the proposed loss term, Multi-camera Negative Loss, is implemented in class DistanceLoss. We also provide training data (DukeSCT and MarketSCT in paper) information in duke_sct.txt and market_sct.txt. 

The code is based on [Cysu/open-reid](https://github.com/Cysu/open-reid).  

## Environment

The code has been tested on Pytorch 1.0.0 and Python 3.7. 

Other required packages: fire, protobuf, tensorboardX.

We use ResNet-50 as the backbone. A pretrained model file is needed. Please put [this file](https://download.pytorch.org/models/resnet50-19c8e357.pth) in the repo directory. 

## Dataset Preparation 

**1. Download Market-1501 and DukeMTMC-reID**

**2. Make new directories in data and organize them as follows:**
<pre>
+-- data
|   +-- market
|       +-- bounding_box_train_sct
|       +-- query
|       +-- boudning_box_test
|   +-- duke
|       +-- bounding_box_train_sct
|       +-- query
|       +-- boudning_box_test
</pre>

**3. Copy images to the above directories.**
 
Training images used for SCT datasets are listed in market.txt and duke.txt. Query and gallery images remain the same. 

## Train and test

To train with our proposed MCNL or Triplet baseline, simply run train.sh. 

## Citation

If you find this code useful, please kindly cite the following paper:


    @inproceedings{zhang2020single,
        title={Single Camera Training for Person Re-identification},
        author={Zhang, Tianyu and Xie, Lingxi and Wei, Longhui and Zhang, Yongfei and Li, Bo and Tian, Qi},
        booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
        year={2020}
        }
