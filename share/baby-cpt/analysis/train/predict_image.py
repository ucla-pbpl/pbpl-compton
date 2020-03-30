import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import preprocess_image
import train_image
from mpl_toolkits.mplot3d import Axes3D
import argparse
import toml

def main():
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
    
    args = parser.parse_args()

    dirname = args.dir 
    name = args.h5_name
    filename = dirname+'/'+name+'.h5'
    #
    #filename = 'test-right-y/varY-col-2e6-UPFZET-0.h5'

    f = h5py.File(filename, 'r')
    truth = np.zeros([50, 31])
    #name = os.path.splitext(filename)[0]

    try:
        with np.load(dirname+'/'+name+'.npz') as data:
            truth = data['histo']
    except (IOError):
        truth = None

    points = np.array(f['position'])
    energy = np.array(f['edep'])

    conf = toml.load(args.config)
    x_max = int(conf['Simulation']['XMax'])
    x_bins = int(conf['Simulation']['XBins'])
    y_max = int(conf['Simulation']['YMax'])
    y_bins = int(conf['Simulation']['YBins'])
    l_y_bins = int(conf['PrimaryGenerator']['YBins'])
    l_e_bins = int(conf['PrimaryGenerator']['EBins'])
    b_out = preprocess_image.get_image(points, energy, x_bins, x_max, y_bins, y_max)

    model = train_image.build_model(x_bins, y_bins, l_y_bins, l_e_bins)

    checkpoint_path = args.model#"models/col-right-y-mixed-startover-cp.ckpt"

    # Loads the weights
    model.load_weights(checkpoint_path)

    test_predictions = model.predict([[b_out]])


    fig3 = plt.figure(constrained_layout=True)
    gs = fig3.add_gridspec(2, 2)
    #fig, ax = plt.subplots(2, 2)
    if truth is not None:
        f3_ax1 = fig3.add_subplot(gs[0, 1])
        truth_im = f3_ax1.imshow(truth)
        f3_ax1.set_title("truth")
        f3_ax1.set_xlabel('E')
        f3_ax1.set_ylabel('Y')
        #fig3.colorbar(truth_im, cax=f3_ax1)

    f3_ax2 = fig3.add_subplot(gs[1, 1])
    pre_im = f3_ax2.imshow(test_predictions.reshape(l_y_bins, l_e_bins))
    f3_ax2.set_title("prediction")
    f3_ax2.set_xlabel('E')
    f3_ax2.set_ylabel('Y')
    #fig.colorbar(pre_im, cax=f3_ax2)

    f3_ax3 = fig3.add_subplot(gs[:, 0])
    f3_ax3.set_title('gs[:, 0]')
    im = f3_ax3.imshow(b_out, interpolation='bilinear', origin='lower')
    #ax.clabel(CS, inline=1, fontsize=10)
    f3_ax3.set_title("e dep")
    f3_ax3.set_xlabel('Y')
    f3_ax3.set_ylabel('X')
    #fig.colorbar(im, cax=ax[1,0])
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(truth_im, cax=cbar_ax)

    plt.savefig("compare_image_test-"+name+".png")

if __name__ == "__main__":
   main()
