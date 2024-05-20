# DyAtGNN
DyAtGNN: Dynamic Attention Graph Neural Networks for Dynamic Graph

This repository contains implementations for the DyAtGNN paper.

### How to get the preprocessed datasets
Please see the `./data/dataset_preprocess.py` with all the datasets in paper.

### How to run this experiments:
```bash
python3 train.py --data $data --batch 10 --units 32  --lr 0.001 --wd 0.0001
```
