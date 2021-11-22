# Cross-dataset Generalization


## Setting up

Set up a conda environment in the following way.

`conda create -n myenv python=3.8 scipy=1.5.2`

`conda install pytorch torchvision cudatoolkit=11.3 -c pytorch`

`conda install -c conda-forge matplotlib`

`conda install -c conda-forge opencv`

`pip install torchsummary`

`conda install -c conda-forge scikit-learn`

`conda install av -c conda-forge`

`conda install -c conda-forge ipdb`

`conda install -c conda-forge prettytable`

`conda install pytorch-lightning -c conda-forge`

`conda install -c anaconda pandas`

`conda install -c conda-forge tqdm`

You also will want a wandb-account to keep track of your experiments.

`pip install wandb`

#### Download the dataset
The Diving48 dataset is available for download [here](http://www.svcl.ucsd.edu/projects/resound/dataset.html).
The shape and texture versions will be made available soon, please contact the repository owner if you want them earlier.

#### Modify config file to include the correct data paths
In the configuration files (located under `configs/`), modify the
- path to data: `data_folder`
- path to JSONs: `json_data_train`, `json_data_val`, `json_data_test`

#### How to train from scratch?
Run:

`python main.py --config configs/berzelius_clstm.json --job_identifier 389459 --fast_dev_run=False --log_every_n_steps=5 --gpus=1`

There are also sbatch-scripts for Slurm cluster training under `run_scripts`.

where,
- `config`: is the path to the .json config-file,
- `job_identifier`: should be a unique string for your job to not overwrite checkpoints or other output from the run,
- `fast_dev_run`, `log_every_n_steps`, `gpus`: all communicate with the PyTorch Lightning trainer, see documentation [here](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html).

### Hyperparameters
Please refer to the config files under `configs/`.


## How to use a pre-trained model?
- Insert the path to a `.ckpt` file in the configs, and set `inference_only` to True in the config file.
Run, for example:

`python main.py --config configs/inference_convlstm.json --job_identifier 389459 --fast_dev_run=False --log_every_n_steps=5 --gpus=1`


## LICENSE
The repository was initially forked from a [repository](https://github.com/TwentyBN/smth-smth-baseline/) created by TwentyBN. It has been heavily modified by this repository owner since then, adapting the repository to use PyTorch Lightning. 
Most code is copyright (c) 2018 Twenty Billion Neurons GmbH under an MIT Licence. See the file `LICENSE` for details.
Some code snippets have been taken from Keras (see `LICENSE_keras`) and the PyTorch (see `LICENSE_pytorch`). See comments in the source code for details.
