# GRACE

The official PyTorch implementation of deep GRAph Contrastive rEpresentation learning.

## Dependencies

- torch 1.4.0
- torch-geometric 1.5.0
- sklearn 0.21.3
- numpy 1.18.1
- pyyaml 5.3.1

Install all dependencies using
```
pip install -r requirements.txt
```

If you encounter problems during installing torch-geometric, please refer to the installation manual on its [official website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Usage

Train and evaluate the model on a dataset by executing
```
python train.py --dataset Cora
```
Note that the dataset argument should be one of [ Cora, CiteSeer, PubMed, DBLP ].