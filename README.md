[![Conference](http://img.shields.io/badge/ICCV-2023-4b44ce.svg)](https://iccv2023.thecvf.com)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Paper](http://img.shields.io/badge/paper-arxiv.2212.00767-B31B1B.svg)](https://arxiv.org/abs/2212.00767)
# Exploiting Proximity-Aware Tasks for Embodied Social Navigation
**Enrico Cancelli, Tommaso Campari, Luciano Serafini, Angel X. Chang, Lamberto Ballan**

Accepted at ICCV 2023

Short version: [[CVPR EAI workshop paper](https://embodied-ai.org/papers/2023/11.pdf)]

This repository contains the implementation and episode dataset from the paper.

**Warning:** It is still WIP. Feel free to open an issue if you encounter some problems or have some questions.
## Setup
### Setting up conda 
```bash
conda create -n socialnav python=3.6
conda activate socialnav

pip install cython
```

### Installing the project
Our code is based on habitat-lab 0.1.7 (please refer to the original repo). Clone this repo and install it.
```bash
cd habitat-lab
pip install -e .
```
(plus habitat-sim 0.2.1, see original repo for building options)
```bash
conda install habitat-sim=0.2.1 withbullet headless -c conda-forge -c aihabitat
```
Also install both habitat-lab and habitat-sim requirements

### Getting people's meshes
People's meshes are taken from the original challenge data. To get them, clone the original repository using the following script:
```bash
git clone https://github.com/StanfordVL/iGibsonChallenge2021.git

cd ../iGibsonChallenge2021
./download.sh

mkdir ../iGibson/gibson2/data
mv gibson_challenge_data_2021/* ../iGibson/gibson2/data
```
After download, substitute the `PATH_TO_PEOPLE_MESHES` variable in the `habitat-lab/habitat/sims/igibson_challenge/social_nav.py`
with the correct path (will be made a parameter in future).

### HM3D-S episode dataset
The dataset is in the `habitat-lab/data/dataset/pointnav/hm3d` folder.
To use it add this parameter to your main config file:
```
BASE_TASK_CONFIG_PATH: configs/datasets/pointnav/hm3d.yaml
```

# Examples
TODO: this section is work in progress.
For now you can use the following command with the associated config for the baseline:

`python habitat_baselines/run.py
--exp-config
../baseline.yaml
--run-type
eval (or train)
--name
example
`

and this for te full model:

`python habitat_baselines/run.py
--exp-config
../full.yaml
--run-type
eval (or train)
--name
example
`
Please consult the original repository for details about the config system and habitat-lab and habitat_baselines parameters.
