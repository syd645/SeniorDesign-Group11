# Senior Design Work 
We completed the simulations for senior design using this repositiory. We have edited the files from the original repo to work with the HAR dataset described below as well as to include Differential Privacy. 

### Dataset
In order to run the simulations, the HARdataset needs to be downloaded from the following linK:
     https://github.com/xmouyang/FL-Datasets-for-HAR/tree/main/datasets/HARBox

The data_pre.py file within the dataset then needs to be replaced by the one in this repo. The resulting .pt file then needs to be placed in the main directory of this repo. 

We were not able to include the .pt file we used due to the large size. 

Once this complete you are ready to run simulations. 

### Running Simulations
To run the simulations, you should use the formatting explained in the original README listed below, but with the following arguments:
    --model resent 
    --dataset HAR_LS

Using these arguments you can run the federated learning or the centralized model (main_nn.py, main_nn_DP.py) simulations with or with Differential Privacy (DP) with the following files:
    - Federated Learning: 
        - main_fed.py (No DP)
        - main_fed_DP.py (DP)
    - Centralized:
        - main_nn.py (No DP)
        - main_nn_DP.py (DP)


## Federated Learning [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4321561.svg)](https://doi.org/10.5281/zenodo.4321561)

This is a fork of the federated learning framework found at: https://github.com/shaoxiongji/federated-learning
We have edited this framework to run advanced convolutional neural networks, different datasets as well as to implement differential privacy. 

This is partly the reproduction of the paper of [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)   
Only experiments on MNIST and CIFAR10 (both IID and non-IID) is produced by far.

Note: The scripts will be slow without the implementation of parallel computing. 

## Requirements
python>=3.6  
pytorch>=0.4

## Run

The MLP and CNN models are produced by:
> python [main_nn.py](main_nn.py)

Federated learning with MLP and CNN is produced by:
> python [main_fed.py](main_fed.py)

Federated learning with differential privacy:
> python [main_fed_DP.py](main_fed_DP.py)

Advanced CNN model simulation with differential privacy: 
> python [main_nn_DP.py](main_nn_DP.py)


See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

`--all_clients` for averaging over all client models

NB: for CIFAR-10, `num_channels` must be 3.

## Results
### MNIST
Results are shown in Table 1 and Table 2, with the parameters C=0.1, B=10, E=5.

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP|  94.57%     | 70.44%         |
| FedAVG-CNN|  96.59%     | 77.72%         |

Table 2. results of 50 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP| 97.21%      | 93.03%         |
| FedAVG-CNN| 98.60%      | 93.81%         |


## Ackonwledgements
Acknowledgements give to [youkaichao](https://github.com/youkaichao).

## References
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.

## Cite As
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561


