import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import preprocess_image
import train_image_energy_reusable
from mpl_toolkits.mplot3d import Axes3D
import argparse
import toml
from skimage.transform import resize
import scipy.stats as stats
from pbpl import common
import data_normalization

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
    

    dirname = args.dir 
    name = args.h5_name
    filename = dirname+'/'+name+'.h5'
    #
    #filename = 'test-right-y/varY-col-2e6-UPFZET-0.h5'

    gin = h5py.File(filename, 'r')
    #name = os.path.splitext(filename)[0]


    edep = gin['compton-electron']['edep'][0] # (5, 300, 450)
    xbin = gin['compton-electron']['xbin'][:] #6
    ybin = gin['compton-electron']['ybin'][:] #300
    zbin = gin['compton-electron']['zbin'][:] #450
    print(ybin[0], ybin[-1])
    print(zbin[0], zbin[-1])


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


    assert x_max == zbin[-1], "x/z={} bound doesn't match {}".format(x_max, zbin[-1])
    assert y_max == ybin[-1], "y={} bound doesn't match {}".format(y_max, ybin[-1])

    edep = edep.sum(axis=0)
    #plt
    im = plt.imshow(edep)
    plt.savefig("raw-"+name+".png")

    edep_resized = resize(edep, (y_bins, x_bins))
    print(edep_resized.shape)
    edep_max = 0.1e-9*80#np.max(edep_resized)
    edep_normalized, edep_max = data_normalization.normalize_examples_indi(conf,
        np.array([edep_resized]))

    model = train_image_energy_reusable.build_model(conf)

    checkpoint_path = args.model#"models/col-right-y-mixed-startover-cp.ckpt"

    # Loads the weights
    model.load_weights(checkpoint_path)

    test_predictions = data_normalization.recover_labels_indi(conf,
        model.predict([[edep_normalized[0, int(y_bins/2), np.newaxis]]]), edep_max)

    common.setup_plot()
    fig3 = plt.figure(figsize=(3+3/8, 4+4/8), dpi=600, constrained_layout=True)
    gs = fig3.add_gridspec(nrows=2, ncols=1, height_ratios = [10., 7.])

    f3_ax1 = fig3.add_subplot(gs[1, 0])
    photon_e_bins = np.logspace(np.log(l_e_lower), np.log(l_e_upper), l_e_bins+1, base=np.e)
    print(photon_e_bins)

    try:
        with np.load(dirname+'/'+name+'.npz') as data:
            truth = data['histo']
    except (IOError):
        photon_energies = np.array([1, 1.8, 2, 3])
        photon_number = np.array([1e9, 1e9, 1e9, 1e9])
        width = 0.001
        truth = np.zeros((l_y_bins, l_e_bins))
        for i in range(4):
            ej_l = photon_energies[i]-3*width
            ej_u = photon_energies[i]+3*width
            x = np.linspace(ej_l, ej_u, 100)
            float_weights = stats.norm.pdf(x, photon_energies[i], width/2)*photon_energies[i]*photon_number[i]
            #print(float_weights)
            truth[int(l_y_bins/2)] = truth[int(l_y_bins/2)] + np.interp(photon_e_bins[:-1], x, float_weights)
            

    if truth is not None:
        truth_c = truth[int(l_y_bins/2)]
        #f3_ax1.plot(photon_e_bins[:-1], truth_c, label = "truth")
        f3_ax1.set_title("truth vs prediction")
        
        #f3_ax1_c = fig3.add_subplot(gs[0, 3])
        #fig3.colorbar(truth_im, cax=f3_ax1_c, shrink=0.6)

    prediction = test_predictions.reshape(l_y_bins, l_e_bins)
    prediction_c = prediction[int(l_y_bins/2)]
    f3_ax1.plot(photon_e_bins[:-1], prediction_c.clip(min=0), label = "prediction")
    #pre_im = f3_ax2.imshow(prediction)
    #f3_ax2.set_title("prediction")

    f3_ax1.set_xscale('log')
    f3_ax1.set_xlabel('energy (MeV)')
    f3_ax1.set_ylabel('photon energy density (MeV/MeV)')
    f3_ax1.legend()

    f3_ax3 = fig3.add_subplot(gs[0, 0])
    f3_ax3.set_title('gs[:, 0]')
    #im = f3_ax3.imshow(edep_normalized, interpolation='bilinear', origin='lower')
    #ax.clabel(CS, inline=1, fontsize=10)
    #f3_ax3.set_title("e dep")
    f3_ax3.set_xlabel('Z (mm)')
    f3_ax3.set_ylabel('E (MeV)')
    #plot_e = edep_normalized
    f3_ax3.plot(np.linspace(0, x_max, x_bins+1)[:-1], edep_normalized[0, int(y_bins/2)])
    f3_ax3.set_title("normalized energy deposition")
    #f3_ax3_c = fig3.add_subplot(gs[:, 1])
    #fig3.colorbar(im, cax=f3_ax3_c, shrink=0.6)
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(truth_im, cax=cbar_ax)
    model_folder = os.path.basename(os.path.dirname(checkpoint_path))
    plt.savefig(model_folder+'/'+args.out+"-"+name+".png")
    np.savez(model_folder+'/'+args.out+"-"+name+".npz", prediction=prediction)

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
    args.dir = "fake-data"
    args.h5_name = "darpa"
    main(args)
