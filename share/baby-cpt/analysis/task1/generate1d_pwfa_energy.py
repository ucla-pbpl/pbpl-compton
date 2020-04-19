import argparse
import h5py
import toml
from skimage.transform import resize
import numpy as np
import scipy.stats as stats
import random
import matplotlib.pyplot as plt
import string

def plot_preview(edep, truth, desc):
    edep_resized = edep.T
    fig3 = plt.figure(constrained_layout=True)
    gs = fig3.add_gridspec(2, 2)

    if truth is not None:
        f3_ax1 = fig3.add_subplot(gs[0, 1])
        truth_im = f3_ax1.imshow(truth)
        f3_ax1.set_title("truth")
        f3_ax1.set_xlabel('E')
        f3_ax1.set_ylabel('Y')
        #fig3.colorbar(truth_im, cax=f3_ax1)
        f3_ax1 = fig3.add_subplot(gs[1, 1])
        truth_summed = np.sum(truth, axis = 0)
        f3_ax1.plot(truth_summed)
        f3_ax1.set_title("truth")
        f3_ax1.set_xlabel('E')

    f3_ax3 = fig3.add_subplot(gs[:, 0])
    f3_ax3.set_title('gs[:, 0]')
    im = f3_ax3.imshow(edep_resized, interpolation='bilinear', origin='lower')
    #ax.clabel(CS, inline=1, fontsize=10)
    f3_ax3.set_title("e dep")
    f3_ax3.set_xlabel('Y')
    f3_ax3.set_ylabel('Z')
    fig3.colorbar(im, ax=f3_ax3)

    plt.savefig(desc)

# There are 100 gamma energy bins in the training data.  
# Though the bin energies are logarithmically spaced, 
# the gammas are sampled linearly within each bin.  

def generate_mix(edep, num_events, energy_ranges, x_max, 
    x_bins, y_max, y_bins, 
    l_y_lower, l_y_upper, l_y_bins,
    l_e_lower, l_e_upper, l_e_bins, in_e_bins, in_e, data_file):

    random_affix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    #print(l_e_lower)
    #return
    #ln_e_lower = np.log(l_e_lower)
    #ln_e_upper = np.log(l_e_upper)
    num_events = np.squeeze(np.array(num_events))#100
    print("num_events.shape", num_events.shape)
    energy_ranges = np.squeeze(np.array(energy_ranges))#101
    print("energy_ranges.shape", energy_ranges.shape)
    energy_range_lengths = np.diff(energy_ranges)#100
    energy_midpoints = energy_ranges[0:-1]+energy_range_lengths/2

    resampled_energy_ranges = np.logspace(np.log(l_e_lower), np.log(l_e_upper), l_e_bins+1, base=np.e)

    #edep_flattened = edeps.reshape(len(num_events),-1)
    #edep_per_sheet = np.sum(edep_flattened, axis=1)
    #print(edep_per_sheet.shape)
    #ratios = edep_per_sheet/energy_per_sheet

    edep = np.array(edep).sum(axis=1)
    edep_resized = resize(edep, (len(edep), y_bins, x_bins))
    edep_transposed = edep_resized.transpose(1, 2, 0)

    # the avaraged response of one MeV in the energy range (a, b)
    edep_transposed_averaged = edep_transposed/(num_events*energy_midpoints) #correct
    print(edep_transposed_averaged[0][1][1])

    num_simulations = len(num_events)

    e_weights = np.interp(resampled_energy_ranges[:-1], in_e_bins, in_e)
    weights = np.interp(energy_ranges[:-1], in_e_bins, in_e)#
    ###################### key step ######################
    new_edep = np.dot(edep_transposed_averaged, weights*energy_range_lengths)
    ######################################################
    histo = np.array([e_weights])
    print("save image 0")
    test_output_filename = data_file+"-"+random_affix
    plot_preview(new_edep, histo, test_output_filename+".png")
    with h5py.File(test_output_filename+".h5", "w") as f:
        grp = f.create_group("compton-electron")
        dset = grp.create_dataset("edep", data = [[new_edep]])
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
    #parser.add_argument("--h5", required=True,
    #    help="set where descrete input is.")
    parser.add_argument("--config", required=True,
        help="the config toml for simulation file.")
    parser.add_argument("--out_file", required=True, 
        help="the output training set and testing set. ")
    parser.add_argument("--sample_npz", required=True,
        help="the sample photon energy distribution whose image to generate.")
    
    args = parser.parse_args()
    conf = toml.load(args.config)
    #f["compton-electron']['edep']<HDF5 dataset "edep": shape (100, 1, 5, 300, 450), type "<f4">
    gin = h5py.File('fake-data/response-4500A.h5')
    edeps = gin['compton-electron']['edep'] # (100, 1, 5, 300, 450)
    print(edeps.shape)
    edeps = np.squeeze(edeps)
    xbin = gin['compton-electron']['xbin'][:] #6
    ybin = gin['compton-electron']['ybin'][:] #301
    zbin = gin['compton-electron']['zbin'][:] #451
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

    with np.load(args.sample_npz) as data:
        in_e_bins = data["photon_bins"]
        in_e = data['photon_energy']
    # There are 100 gamma energy bins in the training data.  
    # Though the bin energies are logarithmically spaced, 
    # the gammas are sampled linearly within each bin.  

    generate_mix(edeps, num_events, energy_ranges, x_max, 
    x_bins, y_max, y_bins, 
    l_y_lower, l_y_upper, l_y_bins,
    l_e_lower, l_e_upper, l_e_bins, in_e_bins, in_e, args.out_file)

if __name__ == "__main__":
    main()
