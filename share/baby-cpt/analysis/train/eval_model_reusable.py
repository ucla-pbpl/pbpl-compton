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


def main(config):
    

    common.setup_plot()

    conf = toml.load(config)
    l_y_bins = int(conf['PrimaryGenerator']['YBins'])
    l_e_bins = int(conf['PrimaryGenerator']['EBins'])
    x_bins = int(conf['Simulation']['XBins'])
    y_bins = int(conf['Simulation']['YBins'])

    data_files = conf['NeuralNetwork']["EvalFileNames"]
    
    train_examples, train_labels, test_examples, test_labels, name_string = train_image_energy_reusable.load_data(conf, data_files)

    print("train_examples.shape, train_labels.shape", train_examples.shape, train_labels.shape)

    test_examples, max_ex = data_normalization.normalize_examples_indi(conf,test_examples)
    test_labels_normalized = data_normalization.normalize_labels(conf,test_labels)
    
    model = train_image_energy_reusable.build_model(conf)
    args_out = conf['NeuralNetwork']["ModelName"]
    data_files = conf['NeuralNetwork']["DataFileNames"]
    name_string=""
    name_string=name_string.join(data_files).replace("/", "")
    checkpoint_path = "models-grid"+"-"+args_out+"/"+name_string+".ckpt"
    model.load_weights(checkpoint_path)

    test_predictions =  data_normalization.recover_labels_indi(conf,
                model.predict(test_examples), max_ex)
    mse = ((test_labels - test_predictions)**2).mean()
    print("///////////////////////")
    print("grand mse", mse)
    print("///////////////////////")
    with open("models-grid"+"-"+args_out+"/mse.txt", "w") as f:
        f.write(str(mse))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='eval network based on data file')
    parser.add_argument("--config", required=True, 
        help="set dimensions of input and output data")
    args = parser.parse_args()
    main(args.config)