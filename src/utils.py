import numpy as np
import pathlib
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

def mean_cross_entropy_logits(logits, y):
  probs = softmax(logits, -1)
  y_probs = probs[np.arange(logits.shape[0]), y]
  
  percent_accuracy = (y_probs == probs.max(1)).mean() * 100
  
  return (-np.log(y_probs)).mean(), percent_accuracy