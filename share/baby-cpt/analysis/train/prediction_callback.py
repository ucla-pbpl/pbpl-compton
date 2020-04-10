import predict_summed_image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class EarlyStoppingWhenOutputRepeats(tf.keras.callbacks.Callback):
  """Stop training when the the output for different inputs are too similar.

  Arguments:
      patience: Number of epochs to wait. After this
      number of no improvement, training stops.
  """

  def __init__(self, patience, config_file, slice_index):
    super(EarlyStoppingWhenOutputRepeats, self).__init__()

    self.patience = patience

    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None

    self.config_file = config_file
    self.slice_index = slice_index

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = 0

  def on_epoch_end(self, epoch, logs=None):
    #python ../train/predict_summed_image.py --dir . --h5_name test_saving_h5-901-SQP265 --model models/test_saving_h5.ckpt --config task1_small.toml --out correctly_normalized
    filename  = ["../task1/train-energy-801.h5", "../task1/test-energy-900.h5"]#fake-0-W6STL1.h5 OR train-801.h5
    config_file = self.config_file
    edep_normalized_1, edep_max_1 = predict_summed_image.preprocess_summed_h5(filename[0], config_file)
    edep_normalized_2, edep_max_2 = predict_summed_image.preprocess_summed_h5(filename[1], config_file)
    print(edep_normalized_1.shape)
    edep_plot_1 = edep_normalized_1.T
    edep_plot_2 = edep_normalized_2.T
    test_predictions_1 = (self.model.predict([[edep_plot_1[self.slice_index, np.newaxis]]])).flatten()
    test_predictions_2 = (self.model.predict([[edep_plot_2[self.slice_index, np.newaxis]]])).flatten()

    name_string = "epoch-{}-predictions".format(epoch)
    plt.plot(edep_plot_1[self.slice_index], label = '1-input')
    plt.plot(edep_plot_2[self.slice_index], label = '2-input')
    plt.plot(test_predictions_1, label = "1")
    plt.plot(test_predictions_2, label = "2")
    plt.legend()
    #plt.show()
    plt.savefig(name_string+".png")
    plt.clf()
    weights = self.model.get_weights()
    #print(weights)

    ratio = test_predictions_1/test_predictions_2
    std = np.std(ratio)
    if std >self.best:
      self.best = std
      self.wait = 0
      # Record the best weights if current results is better (less).
      #self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        #print('Restoring model weights from the end of the best epoch.')
        #if(self.best_weights is None):
        #    return
        #self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping due to similar outputs ' % (self.stopped_epoch + 1))