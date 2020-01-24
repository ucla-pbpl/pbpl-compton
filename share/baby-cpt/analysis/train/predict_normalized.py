import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import preprocess_image
import train_normalized
from mpl_toolkits.mplot3d import Axes3D

def main():
    filename = 'test-right-y/gYE-col-2e6-4XKZZU-0.h5'
    #
    f = h5py.File(filename, 'r')
    truth = np.zeros([50, 31])
    name = os.path.splitext(filename)[0]

    with np.load(name+'.npz') as data:
        truth = data['histo']

    points = np.array(f['position'])
    energy = np.array(f['edep'])
    b_out = preprocess_image.get_image(points, energy)

    model = train_normalized.build_model()

    checkpoint_path = "models/col-norm-cp.ckpt"

    # Loads the weights
    model.load_weights(checkpoint_path)

    test_predictions = model.predict([[train_normalized.norm_energy(b_out)]])


    fig, ax = plt.subplots(2, 2)
    truth_im = ax[0, 0].imshow(truth)
    ax[0, 0].set_title("truth")
    ax[0, 0].set_xlabel('E')
    ax[0, 0].set_ylabel('Y')
    #fig.colorbar(truth_im, cax=ax[0, 0])
    pre_im = ax[0, 1].imshow(test_predictions.reshape(truth.shape))
    ax[0, 1].set_title("prediction")
    ax[0, 1].set_xlabel('E')
    ax[0, 1].set_ylabel('Y')
    #fig.colorbar(pre_im, cax=ax[0, 1])

    im = ax[1, 0].imshow(b_out, interpolation='bilinear', origin='lower')
    #ax.clabel(CS, inline=1, fontsize=10)
    ax[1, 0].set_title("e dep")
    ax[1, 0].set_xlabel('X')
    ax[1, 0].set_ylabel('Y')
    #fig.colorbar(im, cax=ax[1,0])
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(truth_im, cax=cbar_ax)


    plt.savefig("compare_normalized_test-gYE-col-2e6-4XKZZU-0.png")

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
