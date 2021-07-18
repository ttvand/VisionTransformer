import numpy as np
import pathlib
from PIL import Image
from scipy.special import softmax


def get_data(max_train, max_test):
  data_folder = pathlib.Path(__file__).parent.absolute().parent / 'Data'
  
  train_x = np.load(data_folder / 'training_images.npy')
  train_y = np.load(data_folder / 'training_labels.npy')
  test_x = np.load(data_folder / 'test_images.npy')
  test_y = np.load(data_folder / 'test_labels.npy')
  
  if max_train is not None:
    train_x = train_x[:max_train]
    train_y = train_y[:max_train]
  if max_test is not None:
    test_x = test_x[:max_test]
    test_y = test_y[:max_test]
  
  return train_x, train_y, test_x, test_y

# Convert tensors on the cpu/gpu to numpy
def to_np(tensor):
  return tensor.detach().cpu().numpy()

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
  
  return mean_cross_entropy_logits(preds, y)

def mean_cross_entropy_logits(logits, y):
  probs = softmax(logits, -1)
  y_probs = probs[np.arange(logits.shape[0]), y]
  
  y_pred = probs.argmax(1)
  percent_accuracy = (y_probs == probs.max(1)).mean() * 100
  
  return (-np.log(y_probs)).mean(), percent_accuracy, y_pred

def store_confusion_matrix(
    x, y_actual, y_pred, epoch, save_folder, border_size=2):
  image_dim = x.shape[1]
  img = 255*np.ones(((image_dim+border_size)*10+border_size,
                     (image_dim+border_size)*10+border_size), dtype=np.uint8)
  for i in range(10):
    for j in range(10):
      match_rows = np.where((y_actual == i) & (y_pred == j))[0]
      if match_rows.size:
        match_row = np.random.choice(match_rows)
        img[border_size*(i+1)+i*image_dim:((i+1)*(image_dim+border_size)),
            border_size*(j+1)+j*image_dim:((j+1)*(image_dim+border_size))] = (
          x[match_row])
      else:
        img[border_size*(i+1)+i*image_dim:((i+1)*(image_dim+border_size)),
            border_size*(j+1)+j*image_dim:((j+1)*(image_dim+border_size))] = 0
    
  save_path = save_folder / ('confusion_matrix_' + str(epoch) + '.jpg')
  im = Image.fromarray(img)
  im.save(save_path)