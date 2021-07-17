import pathlib
import torch

import models
import utils


config = {
    'device': 'cuda',
    'n_workers': 0,
    'batch_size': 32,
    'block_size': 4,
    'n_epochs': 10,
    'lr': 4e-3,
    'scheduler': lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 20, eta_min=1e-7),
    'model_folder': pathlib.Path(__file__).parent.absolute().parent / 'Models',
    
    'config_transformer': {
      'num_layers': 1,
      'd_model': 128,
      'nhead': 8,
      'dim_feedforward': 256,
      'dropout': 0.1,
      }
}

train_x, train_y, test_x, test_y = utils.get_data()
models.Model(config, train_x, train_y, test_x, test_y)