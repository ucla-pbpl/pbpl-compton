import predict_summed_image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class LogCallback(tf.keras.callbacks.Callback):
  """writes training progress to a file

  Arguments:
      file_name: the file to write to
  """

  def __init__(self, filename):
    super(LogCallback, self).__init__()

    self.log_filename = filename

  def on_train_begin(self, logs=None):
    with open(self.log_filename, "w") as f:
        f.write("start training\n")


  def on_epoch_end(self, epoch, logs=None):
    with open(self.log_filename, "a+") as f:
        f.write('Epoch: {} - loss: {:7.2f} - mae: {:7.7f} - mse: {:7.7f} - val_loss: {:7.2f} - val_mae: {:7.7f} - val_mse: {:7.7f}\n'.format(
          epoch, logs['loss'], logs['mae'], logs['mse'],logs['val_loss'], logs['val_mae'], logs['val_mse']))


  def on_train_end(self, logs=None):
    with open(self.log_filename, "a+") as f:
      
      f.write('end training')