import h5py
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import preprocess_image
import train_image
from mpl_toolkits.mplot3d import Axes3D

def main(name):
    #filename = 'test-right-y/gYE-col-2e6-4XKZZU-0.h5'
    #filename = 'test-right-y/dgYrayE-col-2e7-SG84SF-0.h5'
    dirname = 'test-right-y/'
    filename = dirname+name+'.h5'
    #
    #filename = 'test-right-y/varY-col-2e6-UPFZET-0.h5'

    f = h5py.File(filename, 'r')
    truth = np.zeros([50, 31])
    #name = os.path.splitext(filename)[0]

    with np.load(dirname+name+'.npz') as data:
        truth = data['histo']

    points = np.array(f['position'])
    energy = np.array(f['edep'])
    b_out = preprocess_image.get_image(points, energy)

    model = train_image.build_model()

    checkpoint_path = "models/col-right-y-mixed-startover-cp.ckpt"

    # Loads the weights
    model.load_weights(checkpoint_path)

    test_predictions = model.predict([[b_out]])


    #fig, ax = plt.subplots(2, 2)
    fig3 = plt.figure(constrained_layout=True)
    gs = fig3.add_gridspec(2, 2)
    f3_ax1 = fig3.add_subplot(gs[0, 1])
    truth_im = f3_ax1.imshow(truth)
    f3_ax1.set_title("truth")
    f3_ax1.set_xlabel('E')
    f3_ax1.set_ylabel('Y')
    #fig3.colorbar(truth_im, cax=f3_ax1)

    f3_ax2 = fig3.add_subplot(gs[1, 1])
    pre_im = f3_ax2.imshow(test_predictions.reshape(truth.shape))
    f3_ax2.set_title("prediction")
    f3_ax2.set_xlabel('E')
    f3_ax2.set_ylabel('Y')
    #fig.colorbar(pre_im, cax=f3_ax2)

    f3_ax3 = fig3.add_subplot(gs[:, 0])
    f3_ax3.set_title('gs[:, 0]')
    im = f3_ax3.imshow(b_out, interpolation='bilinear', origin='lower')
    #ax.clabel(CS, inline=1, fontsize=10)
    f3_ax3.set_title("e dep")
    f3_ax3.set_xlabel('X')
    f3_ax3.set_ylabel('Y')
    #fig.colorbar(im, cax=ax[1,0])
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(truth_im, cax=cbar_ax)

    plt.savefig("compare_image_test-"+name+"-startover.png")

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main("gYE-col-2e6-4XKZZU-0")
   main("dgYrayE-col-2e7-N5N13U-0")
   main('rYgE-col-2e7-1L1XU4-0')
   main("varY-col-2e6-UPFZET-0")
