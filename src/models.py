import datetime
import numpy as np
from skimage.util import view_as_blocks
import time
import torch
from torch import nn

import utils


# Convert tensors on the cpu/gpu to numpy
def to_np(tensor):
  return tensor.detach().cpu().numpy()

class ZDataset:
  def __init__(self, x, y, block_size):
    # x = np.arange(28*28*2).reshape((-1, 28, 28))
    x_blocks = view_as_blocks(x/255, (1, block_size, block_size))[:, :, :, 0]
    x_blocks = x_blocks.reshape([x.shape[0], -1] + list(x_blocks.shape)[3:])
    x_blocks = x_blocks.reshape(list(x_blocks.shape[:2]) + [-1])
    zero_blocks = np.zeros(
      (x_blocks.shape[0], 1, x_blocks.shape[2]), dtype=x_blocks.dtype)
    x_blocks = np.concatenate([zero_blocks, x_blocks], 1)
    self.x = torch.from_numpy(x_blocks)
    self.y = torch.from_numpy(y)
    self.sample_id = 0
    self.samples_per_epoch = x.shape[0]

  def __len__(self):
    return self.samples_per_epoch

  def __getitem__(self, idx):
    s_id = self.sample_id
    # print(self.sample_id)
    self.sample_id = (self.sample_id + 1) % self.samples_per_epoch
    
    return {
        'x': self.x[s_id],
        'y': self.y[s_id],
    }


class VisionTransformer(nn.Module):
  def __init__(self, config, block_size, device):
    super(VisionTransformer, self).__init__()
    
    self.device = device
    
    self.vision_embedder = nn.Linear(
      block_size*block_size, config['d_model']//2)
    
    self.num_embeddings = int(28*28/(block_size*block_size)) + 1
    self.position_embedder = nn.Embedding(
      self.num_embeddings, config['d_model']//2)
    
    transformer_layer = nn.TransformerEncoderLayer(
      config['d_model'], config['nhead'], config['dim_feedforward'],
      config['dropout'])
    self.transformer = nn.TransformerEncoder(
      transformer_layer, config['num_layers'], norm=None)
    
    self.last_linear = nn.Linear(config['d_model'], 10)

  def forward(self, d):
    B = d['x'].shape[0]
    embedded_vision = self.vision_embedder(d['x'].permute((1, 0, 2)))
    pos_embedding = (self.position_embedder(
      torch.arange(self.num_embeddings, device=self.device)).reshape((
        self.num_embeddings, 1, -1))).repeat(1, B, 1)
    transformed = self.transformer(torch.cat(
      [embedded_vision, pos_embedding], -1))
    output = self.last_linear(transformed[0])
    
    # if np.random.uniform() < 1e-3:
    #   a = to_np(pos_embedding); b = to_np(transformed[0])
    #   import pdb; pdb.set_trace()
    
    return output

def convert_d_types(d, device):
  for k in d.keys():
    d[k] = d[k].to(device).float()
    if k in ['y']:
      d[k] = d[k].to(device).long()
    else:
      d[k] = d[k].to(device).float()
      
def summarize_preds(all_preds, all_y):
  preds = np.concatenate(all_preds)
  y = np.concatenate(all_y)
  
  return utils.mean_cross_entropy_logits(preds, y)

class Model():
  def __init__(self, config, train_x, train_y, test_x, test_y):
    self.config = config
    self.train_x = train_x
    self.train_y = train_y
    self.test_x = test_x
    self.test_y = test_y

  def fit(self):
    self.nn = VisionTransformer(
      self.config['config_transformer'], self.config['block_size'],
      self.config['device'])
    self.nn.to(self.config['device'])

    record_time = str(datetime.datetime.now())[:19]
    model_save_path = self.config['model_folder'] / (record_time + '.pt')

    train_loader = torch.utils.data.DataLoader(
      ZDataset(self.train_x, self.train_y, self.config['block_size']),
      batch_size=self.config['batch_size'],
      num_workers=self.config['n_workers'],
      shuffle=True,
      pin_memory=True)
   
    test_loader = torch.utils.data.DataLoader(
      ZDataset(self.test_x, self.test_y, self.config['block_size']),
      batch_size=self.config['batch_size'],
      num_workers=self.config['n_workers'],
      shuffle=False,
      pin_memory=True)
    
    optimizer_f = lambda par: torch.optim.Adam(par, lr=self.config['lr'])
    optimizer = optimizer_f(self.nn.parameters())
    if self.config.get('scheduler', None) is not None:
      self.scheduler = self.config.get('scheduler')(optimizer)
      
    # Train phase
    met_hist = []
    best_test = float('inf')
    for epoch in range(self.config['n_epochs']):
      print(f"Epoch {epoch+1} of {self.config['n_epochs']}")
      start_time = time.time()
      self.nn.train()
      avg_train_loss = 0
      all_preds = []
      all_y = []
      for batch_id, d in enumerate(train_loader):
        # print(f"Batch id: {batch_id}")
        optimizer.zero_grad()
 
        convert_d_types(d, self.config['device'])
        preds = self.nn(d)
        all_preds.append(to_np(preds))
        batch_y = d['y']
        all_y.append(to_np(batch_y))
   
        loss = nn.CrossEntropyLoss()(preds, batch_y)
        avg_train_loss += to_np(loss) / len(train_loader)
   
        if epoch > 0:
          loss.backward()
        optimizer.step()

      self.nn.eval()
      train_loss, train_acc = summarize_preds(all_preds, all_y)
  
      if self.config.get('scheduler', None) is not None:
        self.scheduler.step()
 
      train_elapsed = time.time() - start_time

      # Test phase
      all_preds = []
      all_y = []
      for batch_id, d in enumerate(test_loader):
        # print(f"Batch id: {batch_id}")
        
        convert_d_types(d, self.config['device'])
        all_preds.append(to_np(self.nn(d)))
        all_y.append(to_np(d['y']))

      test_loss, test_acc = summarize_preds(all_preds, all_y)
      met_hist.append(test_loss)
      if test_loss < best_test:
        best_test = test_loss
        torch.save(
            self.nn.state_dict(),
            model_save_path,
            _use_new_zipfile_serialization=False)
      elapsed = time.time() - start_time
      
      print(f"{epoch+1:3}: {train_loss:8.2f} {test_loss:8.2f}\
{train_acc:8.2f} {test_acc:8.2f} {train_elapsed:8.2f}s {elapsed:8.2f}s")

