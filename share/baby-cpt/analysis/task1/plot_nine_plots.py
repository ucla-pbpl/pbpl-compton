import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import train_image_energy
from mpl_toolkits.mplot3d import Axes3D
import argparse
import toml
from skimage.transform import resize

def preprocess_summed_h5(filename, config_file):
    gin = h5py.File(filename, 'r')
    conf = toml.load(config_file)
    edep = gin['compton-electron']['edep'][0] # (5, 300, 450)
    x_bins = int(conf['Simulation']['XBins'])
    y_bins = int(conf['Simulation']['YBins'])
    #edep = edep.sum(axis=0).T
    edep_resized = resize(edep, (x_bins, y_bins))
    edep_max = np.max(edep_resized)
    edep_normalized = edep_resized/edep_max
    return (edep_normalized, edep_max)

def main(args):
    conf = toml.load(args.config)
    x_max = int(conf['Simulation']['XMax'])
    x_bins = int(conf['Simulation']['XBins'])
    y_max = int(conf['Simulation']['YMax'])
    y_bins = int(conf['Simulation']['YBins'])
    l_y_bins = int(conf['PrimaryGenerator']['YBins'])
    l_e_bins = int(conf['PrimaryGenerator']['EBins'])
    l_y_lower = conf['PrimaryGenerator']['YLower']
    l_e_lower = conf['PrimaryGenerator']['ELower']
    l_y_upper = conf['PrimaryGenerator']['YUpper']
    l_e_upper = conf['PrimaryGenerator']['EUpper']

    model = train_image_energy.build_model(1, x_bins, l_y_bins, l_e_bins)

    checkpoint_path = args.model#"models/col-right-y-mixed-startover-cp.ckpt"

    # Loads the weights
    model.load_weights(checkpoint_path)

    names = ["long_energy_3p-7VA7AA-1600", "long_energy_3p-7VA7AA-400"  
        , "long_energy_g-OISL4N-0"   , "long_energy_g-OISL4N-1700"   
        , "long_energy_random-3A7DNN-200"  
        , "long_energy_random-3A7DNN-1200"]  

    
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=3)
    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            name = names[row*3+col]


            gin = h5py.File(name+".h5", 'r')
            truth = None

            try:
                with np.load(name+'.npz') as data:
                    truth = data['histo']
            except (IOError):
                truth = None


            edep = gin['compton-electron']['edep'][0] # (5, 300, 450)
            xbin = gin['compton-electron']['xbin'][:] #6
            ybin = gin['compton-electron']['ybin'][:] #300
            zbin = gin['compton-electron']['zbin'][:] #450
            print(ybin[0], ybin[-1])
            print(zbin[0], zbin[-1])


            assert x_max == zbin[-1], "x/z={} bound doesn't match {}".format(x_max, zbin[-1])
            assert y_max == ybin[-1], "y={} bound doesn't match {}".format(y_max, ybin[-1])

            edep = edep.sum(axis=0)
            #plt
            #im = plt.imshow(edep)
            #plt.savefig("raw-"+name+".png")

            edep_resized = resize(edep, (y_bins, x_bins))
            edep_max = np.max(edep_resized)
            edep_normalized = edep_resized#/edep_max*10

            test_predictions = model.predict([[edep_normalized[int(y_bins/2), np.newaxis]]])/1.5e-7

            photon_e_bins = np.logspace(np.log(l_e_lower), np.log(l_e_upper), l_e_bins+1, base=np.e)
            #print(photon_e_bins)
            if truth is not None:
                truth_c = truth[int(l_y_bins/2)]
                lp = ax.plot(photon_e_bins[:-1], truth_c, label = "truth")[0]
                
        
                #f3_ax1_c = fig3.add_subplot(gs[0, 3])
                #fig3.colorbar(truth_im, cax=f3_ax1_c, shrink=0.6)

            prediction = test_predictions.reshape(l_y_bins, l_e_bins)
            prediction_c = prediction[int(l_y_bins/2)]
            lt = ax.plot(photon_e_bins, prediction_c, label = "prediction")[0]
            #pre_im = f3_ax2.imshow(prediction)
            #f3_ax2.set_title("prediction")

            ax.set_xscale('log')
            #ax.set_xlabel('E (MeV)')
            #ax.set_ylabel('photons E (MeV)')
            #ax.legend()
    # Create the legend
    fig.legend([lt, lp],     # The line objects
           labels=['truth', 'prediction'],   # The labels for each line
           loc="upper right",   # Position of legend
           borderaxespad=0.1,    # Small spacing around legend box
           )
    fig.suptitle("Model Performance in Testing Sets")
    plt.savefig(args.out+".png")
    #np.savez(args.out+"-"+name+".npz", prediction=prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict gamma spectra based on spectrometer data.')
    parser.add_argument("--model", required=True,
        help="path to saved tensorflow model")
    parser.add_argument("--config", required=True,
        help="the config toml file")
    parser.add_argument("--out", required=True,
        help="output file name suffix")

    args = parser.parse_args()
    main(args)
