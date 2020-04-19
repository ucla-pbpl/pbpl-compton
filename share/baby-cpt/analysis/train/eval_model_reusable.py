import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import train_image_energy_reusable
import predict_summed_image_reusable
from mpl_toolkits.mplot3d import Axes3D
import argparse
import toml
from skimage.transform import resize
from pbpl import common
import data_normalization


def main():
    parser = argparse.ArgumentParser(
        description='eval network based on data file')
    parser.add_argument("--data_file_name", required=True, nargs="+",
        help="set where the training data comes from.")
    parser.add_argument("--config", required=True, 
        help="set dimensions of input and output data")
    parser.add_argument("--out", required=True, 
        help="output suffix")
    parser.add_argument("--model", required=True,
        help="path to saved tensorflow model")
    args = parser.parse_args()

    common.setup_plot()

    conf = toml.load(args.config)
    l_y_bins = int(conf['PrimaryGenerator']['YBins'])
    l_e_bins = int(conf['PrimaryGenerator']['EBins'])
    x_bins = int(conf['Simulation']['XBins'])
    y_bins = int(conf['Simulation']['YBins'])

    data_files = args.data_file_name
    
    train_examples, train_labels, test_examples, test_labels, name_string = train_image_energy_reusable.load_data(conf, data_files)

    print("train_examples.shape, train_labels.shape", train_examples.shape, train_labels.shape)

    test_examples, max_ex = data_normalization.normalize_examples_indi(test_examples)
    test_labels = data_normalization.normalize_labels(test_labels)
    
    model = train_image_energy_reusable.build_model(1, x_bins, l_y_bins, l_e_bins)
    checkpoint_path = args.model#"models/col-right-y-mixed-startover-cp.ckpt"
    # Loads the weights
    model.load_weights(checkpoint_path)

    test_predictions =  data_normalization.recover_labels_indi(
                model.predict(test_examples), max_ex)
    mse = ((test_labels - test_predictions)**2).mean()
    print("///////////////////////")
    print("grand mse", mse)
    print("///////////////////////")

if __name__ == "__main__":
    main()