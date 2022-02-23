# AKGNN
The source code for Adaptive Kernel Graph Neural Network at AAAI2022 (url: https://arxiv.org/abs/2112.04575).

Please cite our paper if you think our work is helpful to you:

```
@inproceedings{ju2022akgnn,
  title={Adaptive Kernel Graph Neural Network},
  author={Ju, Mingxuan and Hou, Shifu and Fan, Yujie and Zhao, Jianan and Ye, Yanfang and Zhao, Liang},
  booktitle={36th AAAI Conference on Artificial Intelligence (AAAI)},
  year={2022}
}
```

## Requirements
* Python 3.8.3
* Please install other pakeages by 
``` pip install -r requirement.txt```

## Usage Example
* Running on Cora:
```python train_cora.py ```
* Running on Citeseer:
```python train_citeseer.py ```
* Running on Pubmed:
```python train_pubmed.py ```

## Results

Our model achieves the following accuracies on Cora, CiteSeer and Pubmed with the public splits:

| Model name   |   Cora    |  CiteSeer |  Pubmed   |
| ------------ | --------- | --------- | --------- |
| AKGNN        |   84.8%   |    73.5%  |   80.4%   |

## Running Environment 

The experimental results reported in paper are conducted on a single NVIDIA GeForce RTX 2080 Ti with CUDA 11.1, which might be slightly inconsistent with the results induced by other platforms.
