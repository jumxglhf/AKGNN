# AKGNN
The source code for Adaptive Kernel Graph Neural Network at AAAI2022

## Requirements
* Python 3.8.3
* Please install other pakeages by 
``` pip install -r requirements.txt```

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
