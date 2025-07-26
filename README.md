# Purge-Gate: Backpropagation-Free Test-Time daptation for Point Clouds via Token purging


This code is mainly based on MATE code base: https://github.com/jmiemirza/MATE


## Data and pre trained preparation
For data preparation please refer to the MATE github. After preparing the datasets, set the address in the config files both for datasets and the 

Please download the pre-trained weight from MATE github (Just source only weights) and place them in the checkpoints directory

## Environment
Instal the environment using these commonds:
conda create -n tta_purge python=3.10
conda activate tta_purge
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt


pip install -U 'git+https://github.com/facebookresearch/iopath'
pip install "git+https://github.com/facebookresearch/pytorch3d.git"


## Reproducing the results
please run:
export CUDA_VISIBLE_DEVICES=<gpu_id>
bash commands.sh

to reproduce all results.