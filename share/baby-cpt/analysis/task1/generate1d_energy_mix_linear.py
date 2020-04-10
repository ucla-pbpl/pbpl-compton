import argparse
import h5py
import toml
from skimage.transform import resize
import numpy as np
import scipy.stats as stats
import random
import matplotlib.pyplot as plt
import string

def generate_distrib(lower, upper, stats_func, num, param=[]):
    x = np.linspace(lower, upper, num)
    xr = stats_func.pdf(x, *param)
    return xr

def get_weights_gaussian(num_simulations, sigma):
    mu = random.randint(0, num_simulations)
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
    num_events = np.squeeze(np.array(num_events))
    print("num_events.shape", num_events.shape)
    energy_ranges = np.squeeze(np.array(energy_ranges))
    print("energy_ranges.shape", energy_ranges.shape)
    energy_range_lengths = np.diff(energy_ranges)
    energy_midpoints = energy_ranges[0:-1]+energy_range_lengths/2

    edep = np.array(edep).sum(axis=1)
    edep_resized = resize(edep, (len(edep), y_bins, x_bins))
    edep_transposed = edep_resized.transpose(1, 2, 0)
    # the avaraged response of one MeV in the energy range (a, b)
    edep_transposed_averaged = edep_transposed/(num_events*energy_midpoints)
    print(edep_transposed_averaged[0][1][1])

    train_data = np.empty([1, y_bins, x_bins])
    train_labels = np.array([np.linspace(0, 31, l_e_bins*l_y_bins)])
    test_data = np.empty([1, y_bins, x_bins])
    test_labels = np.array([np.linspace(0, 31, l_e_bins*l_y_bins)])

    num_simulations = len(num_events)

    for i in range(0, num):
        
        e_component_weights = np.stack((get_weights_gaussian(l_e_bins, 4),
                            get_weights_gaussian(l_e_bins, 7),
                            get_weights_gaussian(l_e_bins, 10), 
                            #get_weights_cosine(l_e_bins)*0.1, 
                            get_weights_random(l_e_bins)*0.1, 
                            #get_weights_gaussian(l_e_bins, 4), 
                            ))
        function_weights = np.random.rand(4)
        e_float_weights = ((np.dot(e_component_weights.T, function_weights)).T)*1#1e9
        e_weights = e_float_weights#.astype(int)
        #print(e_weights)
        weights = np.zeros(num_simulations)#, dtype = int
        for j in range(0, l_e_bins):
            ej_l = l_e_lower+(l_e_upper-l_e_lower)/l_e_bins*(j)
            ej_u = l_e_lower+(l_e_upper-l_e_lower)/l_e_bins*(j+1)
            ej = np.random.uniform(low=ej_l, high=ej_u)
            #print(ej)
            idx_l = np.where(energy_ranges<ej)
            idx_u = np.where(energy_ranges>ej)
            if(len(idx_l)==0 or len(idx_u)==0):
                continue
            else:
                #print("idx[-1][-1]", idx_l[-1][-1], idx_u[0])
                weights[idx_l[-1][-1]]+=e_weights[j]
        #print(weights)
        #return
        #edep_transposed = edep_resized.transpose(2, 3, 1)
        ###################### key step ######################
        new_edep = np.dot(edep_transposed_averaged, weights)
        ######################################################
        #total = int(np.sum(weights))
        #print(total)
        #ys = np.zeros(total)
        #es = np.zeros(total)
        #start = 0
        #for k in range(0, len(weights)):
        #    es[start:start+weights[k]] = (np.random.uniform(low=energy_ranges[k], 
        #    high=energy_ranges[k+1], size=weights[k]))
        #    #(energy_ranges[k]+energy_ranges[k+1])/2
        #    # #np.log
        #    start = start+weights[k]

        #assert ys.shape == es.shape, "bug generating particle es and ys"
        #histo, _, _ = np.histogram2d(ys, es, 
        #    range=[[l_y_lower, l_y_upper],[l_e_lower, l_e_upper]], 
        #    bins=[l_y_bins, l_e_bins])
        histo = np.array([e_weights])
        #print(histo)
        #return
        #ye = np.column_stack((ys, es))
        if (i%100 == 0 or i%100 == 1):
            print("save image "+ str(i))
            test_output_filename = data_file+"-"+random_affix+"-"+str(i)
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
        if (i%10 == 0): 
            test_data = np.append(test_data, [new_edep], axis=0)
            test_labels = np.append(test_labels, [histo.flatten()], axis = 0)
        else:
            train_data = np.append(train_data, [new_edep], axis=0)
            train_labels = np.append(train_labels, [histo.flatten()], axis = 0)
            print("train_labels.shape", train_labels.shape)

    print("train_data.shape", train_data.shape)
    train_data = train_data[1:].astype(float)
    train_labels = train_labels[1:].astype(float)
    test_data = test_data[1:].astype(float)
    test_labels = test_labels[1:].astype(float)
    np.savez(data_file, train_data=train_data, train_labels=train_labels, 
            test_data=test_data, test_labels=test_labels)
    print('done saving '+data_file)
    

def main():
    parser = argparse.ArgumentParser(
        description='generates linear combinations of descrete data files from a h5 file for 1d simulation.')
    parser.add_argument("--h5", required=True,
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
    gin = h5py.File(args.h5)
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
