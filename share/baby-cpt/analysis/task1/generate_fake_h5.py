import argparse
import h5py
import toml
from skimage.transform import resize
import numpy as np
import scipy.stats as stats
import random
import matplotlib.pyplot as plt
import string
import generate1d_mix

# There are 100 gamma energy bins in the training data.  
# Though the bin energies are logarithmically spaced, 
# the gammas are sampled linearly within each bin.  

def generate_ones(edep, num_events, energy_ranges, x_max, 
    x_bins, y_max, y_bins, 
    l_y_lower, l_y_upper, l_y_bins,
    l_e_lower, l_e_upper, l_e_bins, num, data_file):

    for i in range(0, num):
        ###################### key step ######################
        new_edep = np.ones((x_bins, y_bins))
        ######################################################
        histo = np.ones((l_y_bins, l_e_bins))
        if (i%100 == 0 or i%100 == 1):
            print("save image "+ str(i))
            random_affix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            test_output_filename = data_file+"-"+str(i)+"-"+random_affix
            generate1d_mix.plot_preview(new_edep, histo, test_output_filename+".png")
            with h5py.File(test_output_filename+".h5", "w") as f:
                grp = f.create_group("compton-electron")
                dset = grp.create_dataset("edep", data = [[new_edep.T]])
                #dset[0][0] = new_edep
                dset = grp.create_dataset("xbin", (2,), data = [0,0])
                #dset = [0,0]
                dset = grp.create_dataset("ybin", (y_bins,), 
                    data = np.linspace(-y_max, y_max, y_bins))
                #dset = np.linspace(-y_max, y_max, y_bins)
                dset = grp.create_dataset("zbin", (x_bins,), 
                    data = np.linspace(0, x_max, x_bins))
                #dset = np.linspace(0, x_max, x_bins)
            np.savez(test_output_filename+".npz", histo=histo)

    print('done saving '+data_file)
    

def main():
    parser = argparse.ArgumentParser(
        description='generates linear combinations of descrete data files from a h5 file for 1d simulation.')
    parser.add_argument("--h5_name", required=True,
        help="set where descrete input is.")
    parser.add_argument("--config", required=True,
        help="the config toml for simulation file.")
    parser.add_argument("--num", required=True, 
        help="the number of h5 and npz pair should output. ")
    parser.add_argument("--out_file", required=True, 
        help="the output training set and testing set. ")
    
    args = parser.parse_args()
    conf = toml.load(args.config)
    #f["compton-electron']['edep']<HDF5 dataset "edep": shape (100, 1, 5, 300, 450), type "<f4">
    gin = h5py.File(args.h5_name)
    edeps = gin['compton-electron']['edep'] # (100, 1, 5, 300, 450)
    print(edeps.shape)
    edeps = np.squeeze(edeps)
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

    assert x_max == zbin[-1], "x/z bound doesn't match"
    assert y_max == ybin[-1], "y bound doesn't match"

    num_events = gin["num_events"] #100
    energy_ranges = gin['i0'] #101

    # There are 100 gamma energy bins in the training data.  
    # Though the bin energies are logarithmically spaced, 
    # the gammas are sampled linearly within each bin.  

    generate_ones(edeps, num_events, energy_ranges, x_max, 
    x_bins, y_max, y_bins, 
    l_y_lower, l_y_upper, l_y_bins,
    l_e_lower, l_e_upper, l_e_bins, int(args.num), args.out_file)

if __name__ == "__main__":
    main()
