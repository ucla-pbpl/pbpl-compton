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
from pbpl import common
import data_normalization


# (py37) rodan:~/pbpl-compton/share/baby-cpt/analysis/task1/models-selected-Emily-Lisa-Rose-Adam-Bella-THALHEIMER> 
# python ../../train/predict_summed_image_reusable.py --h5_name pwfa --dir ../fake-data --config ../config/task1-Emily-Lisa-Rose-Adam-Bella-THALHEIMER.toml --out jan29 --model ./long_ht_n_glong_ht_n_3plong_ht_n_7g.ckpt

#(py37) rodan:~/pbpl-compton/share/baby-cpt/analysis/nonstripe4500/models-grid-Emily-Lisa-Rose-Adam-Bella-THABIT>
# python ../../train/predict_summed_image_reusable.py --h5_name pwfa5e10-log-cropped --dir ../fake-data --config ../config/nonstripe4500-Emily-Lisa-Rose-Adam-Bella-THABIT.toml --out Mar10 --model ./long_ht_n_glong_ht_n_3plong_ht_n_7g.ckpt
# python ../../train/predict_summed_image_reusable.py --h5_name pwfa5e10-log-cropped --dir ../fake-data --config ../config/nonstripe4500-Emily-Lisa-Rose-Adam-Bella-THABIT.toml --out Mar10-3p2 --model ./long_ht_n_glong_ht_n_3p2long_ht_n_7g.ckpt

# (py37) rodan:~/pbpl-compton/share/baby-cpt/analysis/nonstripe4500/models-grid-THABIT-Copy>
# python ../../train/predict_summed_image_reusable.py --h5_name pwfa5e10-log-cropped --dir ../fake-data --config ../config/THABIT-Copy.toml --out Mar10 --model ./long_ht_n_glong_ht_n_3p2long_ht_n_7g.ckpt

# python ../../train/predict_summed_image_reusable.py --lower_cutoff 2e-1 --out Mar17_bigger_target --h5_name pwfa5e10-log-cropped --dir ../fake-data --config ../config/THABIT-Copy.toml --model ./long_ht_n_glong_ht_n_3p2long_ht_n_7g.ckpt
# python ../../train/predict_summed_image_reusable.py --lower_cutoff 2e-1 --out April25_bigger_target --h5_name pwfa5e10-log-cropped --dir ../fake-data --config ../config/THABIT-Copy.toml --model ./long_ht_n_glong_ht_n_3p2long_ht_n_7g.ckpt

# (py37) rodan:~/pbpl-compton/share/baby-cpt/analysis/nonstripe4500/models-grid-THABIT-Copy-inverse>
# python ../../train/predict_summed_image_reusable.py --h5_name pwfa5e10-log-cropped --dir ../fake-data --config ../config/THABIT-Copy-inverse.toml --out MApril14 --model ./long_inv_3plong_inv_7g.ckpt

# (py37) rodan:~/pbpl-compton/share/baby-cpt/analysis/nonstripe4500/models-grid-THABIT-Copy-walk>
# python ../../train/predict_summed_image_reusable.py --h5_name pwfa5e10-log-cropped --dir ../fake-data --config ../config/THABIT-Copy-walk.toml --out April25 --model ./walk_3000_1.ckpt --lower_cutoff 2e-1
# python ../../train/predict_summed_image_reusable.py --h5_name darpa --dir ../fake-data --config ../config/THABIT-Copy-walk.toml --out May2-walk --model ./walk_3000_1.ckpt --lower_cutoff 2e-1

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
    truth = np.zeros([50, 31])
    #name = os.path.splitext(filename)[0]

    try:
        with np.load(dirname+'/'+name+'.npz') as data:
            truth = data['histo']
    except (IOError):
        truth = None


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
    #fig3 = plt.figure(figsize=(3+3/8, 4+4/8), dpi=600, constrained_layout=True)
    #gs = fig3.add_gridspec(nrows=2, ncols=1, height_ratios = [10., 7.])
    fig3 = plt.figure(figsize=(3+3/8, 2+2/8), dpi=600, constrained_layout=True)
    gs = fig3.add_gridspec(nrows=1, ncols=1)

    f3_ax1 = fig3.add_subplot(gs[0, 0])
    photon_e_bins = np.logspace(np.log(l_e_lower), np.log(l_e_upper), l_e_bins+1, base=np.e)
    print(photon_e_bins)
    

    prediction = test_predictions.reshape(l_y_bins, l_e_bins)
    prediction_c = prediction[int(l_y_bins/2)]
    f3_ax1.plot(photon_e_bins[:-1], prediction_c.clip(min=0), label = "prediction")
    #pre_im = f3_ax2.imshow(prediction)
    #f3_ax2.set_title("prediction")
    if truth is not None:
        truth_c = truth[int(l_y_bins/2)]
        f3_ax1.plot(photon_e_bins[:-1], truth_c, label = "truth")
        #f3_ax1.set_title("truth vs prediction")
        
        max_truth = np.max(truth_c)
        truth_c_normalized = truth_c/max_truth
        prediction_normalized = prediction_c.clip(min=0)/max_truth
        RMSE = np.sqrt(np.mean((prediction_normalized-truth_c_normalized)**2))
        print(f"---------------max truth: {max_truth}, RMSE: {RMSE}---------------")
        #---------------max truth: 1118978413.8537161, RMSE: 0.11553199924011477---------------

    f3_ax1.yaxis.grid(True, which='major')
    f3_ax1.set_xscale('log')
    f3_ax1.set_xlim(left=args.lower_cutoff)
    f3_ax1.set_yscale('log')
    f3_ax1.set_xlabel('energy (MeV)')
    f3_ax1.set_ylabel('photon energy density (MeV/MeV)')
    f3_ax1.legend()
    
    '''
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
    '''
    model_folder = os.path.basename(os.path.dirname(checkpoint_path))
    plt.savefig(model_folder+'/'+args.out+"-"+name+".png")
    np.savez(model_folder+'/'+args.out+"-"+name+".npz", prediction=prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict gamma spectra based on spectrometer data.')
    parser.add_argument("--h5_name", required=True,
        help="set where spectrometer output is.")
    parser.add_argument("--dir", required=True,
        help="set where the h5 file is in")
    parser.add_argument("--model", required=True,
        help="path to saved tensorflow model")
    parser.add_argument("--config", required=True,
        help="the config toml file")
    parser.add_argument("--out", required=True,
        help="output file name suffix")
    parser.add_argument("--lower_cutoff", type=float,
        help="set the lower bound for the plot.")
    
    args = parser.parse_args()
    main(args)