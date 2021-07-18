import pathlib
import torch

import models
import utils


config = {
    'device': 'cuda',
    'n_workers': 0,
    'batch_size': 32,
    'block_size': 4,
    'n_epochs': 50,
    'lr': 2e-4,
    'scheduler': [lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-7), None][1],
    'model_folder': pathlib.Path(__file__).parent.absolute().parent / 'Models',
    
    'config_transformer': {
      'num_layers': 6,
      'd_model': 64,
      'nhead': 8,
      'dim_feedforward': 256,
      'dropout': 0.2,
      },
    
    'max_train': [10000, None][1],
    'max_test': [100, None][1],
}

train_x, train_y, test_x, test_y = utils.get_data(
  config['max_train'], config['max_test'])
  
m = models.Model(config, train_x, train_y, test_x, test_y)
m.fit()