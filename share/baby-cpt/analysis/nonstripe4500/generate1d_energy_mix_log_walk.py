import argparse
import h5py
import toml
from skimage.transform import resize
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
import random
import matplotlib.pyplot as plt
import string

# python generate1d_energy_mix_log_walk.py --config config/THABIT-Copy-walk.toml --num 3000 --out_file walk_3000_1

def generate_distrib(lower, upper, stats_func, num, param=[]):
    x = np.linspace(lower, upper, num)
    xr = stats_func.pdf(x, *param)
    return xr

def get_weights_gaussian(num_simulations, sigma):
    mu = random.randint(int(0-sigma*2), int(num_simulations+sigma*2))
    #variance = 60
    #sigma = 40
    x = np.linspace(0, num_simulations, num_simulations)
    float_weights = stats.norm.pdf(x, mu, sigma)
    return float_weights

def get_weights_random(num_simulations):
    return np.random.rand(num_simulations)
    
def get_weights_cosine(num_simulations):
    peak = random.randint(0, num_simulations)
    return generate_distrib(0, num_simulations, stats.cosine, num_simulations, [peak])

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

def bounded_diffusion(N, num_samples, kick_factor=0.01, drag_factor=0.95):
    var_v_eq = 1/12*kick_factor**2/(1/drag_factor**2-1)
    x = [np.random.rand(num_samples)-0.5]
    v = [np.random.normal(size = num_samples)*var_v_eq**(1/2)]
    for i in range(N-1):
        v.append(v[-1] + kick_factor * (np.random.rand(num_samples) - 0.5))
        v[-1] *= drag_factor
        x.append(x[-1] + v[-1])
        too_small = x[-1] < -0.5
        x[-1][too_small] = -1-x[-1][too_small]
        v[-1][too_small] *= -1
        too_large = x[-1] > 0.5
        x[-1][too_large] = 1-x[-1][too_large]
        v[-1][too_large] *= -1
    x = np.asarray(x)
    #v = np.asarray(v)
    return x

# There are 100 gamma energy bins in the training data.  
# Though the bin energies are logarithmically spaced, 
# the gammas are sampled linearly within each bin.  

def generate_mix(edep, num_events, energy_ranges, x_max, 
    x_bins, y_max, y_bins, 
    l_y_lower, l_y_upper, l_y_bins,
    l_e_lower, l_e_upper, l_e_bins, num, data_file):

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
    print("edep_transposed_averaged.shape", edep_transposed_averaged.shape)

    train_data = np.empty([1, y_bins, x_bins])
    train_labels = np.array([np.linspace(0, 31, l_e_bins*l_y_bins)])
    test_data = np.empty([1, y_bins, x_bins])
    test_labels = np.array([np.linspace(0, 31, l_e_bins*l_y_bins)])

    #num_simulations = len(num_events)

    x = 0.5 + bounded_diffusion(l_e_bins, num)
    print(x.shape)
    print(resampled_energy_ranges.shape)
    histo_f = interp1d(resampled_energy_ranges[:-1], x, axis=0, fill_value="extrapolate")
    histo = histo_f(energy_ranges[:-1])
    print("histo", histo.shape)
    print("histo*lengths", (histo.T*energy_range_lengths).shape)
    new_edep = np.dot(edep_transposed_averaged, (histo.T*energy_range_lengths).T)
    print("new_edep", new_edep.shape)
    new_edep = new_edep.transpose(2, 0, 1)

    test_data = new_edep[:num//10]
    test_labels = x[np.newaxis, :, :num//10]
    # TODO
    test_labels = test_labels.transpose(2, 0, 1)

    train_data = new_edep[num//10:]
    train_labels = x[np.newaxis, :, num//10:]
    train_labels = train_labels.transpose(2, 0, 1)

    print("save image "+ str(0))
    test_output_filename = data_file+"-"+random_affix+"-"+str(0)
    print("test_data[0]", test_data[0].shape)
    print("test_labels[0]", test_labels[0].shape)
    plot_preview(test_data[0], test_labels[0], test_output_filename+".png")
    with h5py.File(test_output_filename+".h5", "w") as f:
        grp = f.create_group("compton-electron")
        dset = grp.create_dataset("edep", data = [[new_edep[0]]])
        #dset[0][0] = new_edep
        dset = grp.create_dataset("xbin", (2,), data = [0,0])
        #dset = [0,0]
        dset = grp.create_dataset("ybin", (y_bins,), 
            data = np.linspace(-y_max, y_max, y_bins))
        #dset = np.linspace(-y_max, y_max, y_bins)
        dset = grp.create_dataset("zbin", (x_bins,), 
            data = np.linspace(0, x_max, x_bins))
        #dset = np.linspace(0, x_max, x_bins)
    np.savez(test_output_filename+".npz", histo=histo[0])

    print("train_data.shape", train_data.shape)
    train_data = train_data[1:].astype(float)
    train_labels = train_labels[1:].astype(float)
    test_data = test_data[1:].astype(float)
    test_labels = test_labels[1:].astype(float)
    np.savez(data_file, train_data=train_data, train_labels=np.squeeze(train_labels), 
            test_data=test_data, test_labels=np.squeeze(test_labels))
    print('done saving '+data_file)
    

def main():
    parser = argparse.ArgumentParser(
        description='generates linear combinations of descrete data files from a h5 file for 1d simulation.')
    #parser.add_argument("--h5", required=True,
    #    help="set where descrete input is.")
    parser.add_argument("--config", required=True,
        help="the config toml for simulation file.")
    parser.add_argument("--num", required=True, 
        help="the number of h5 and npz pair should output. ")
    parser.add_argument("--out_file", required=True, 
        help="the output training set and testing set. ")
    
    args = parser.parse_args()
    conf = toml.load(args.config)
    #f["compton-electron']['edep']<HDF5 dataset "edep": shape (100, 1, 5, 300, 450), type "<f4">
    gin = h5py.File('fake-data/4500.h5')
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

    # There are 100 gamma energy bins in the training data.  
    # Though the bin energies are logarithmically spaced, 
    # the gammas are sampled linearly within each bin.  

    generate_mix(edeps, num_events, energy_ranges, x_max, 
    x_bins, y_max, y_bins, 
    l_y_lower, l_y_upper, l_y_bins,
    l_e_lower, l_e_upper, l_e_bins, int(args.num), args.out_file)

if __name__ == "__main__":
    main()
