import argparse
import h5py
import toml
from skimage.transform import resize
from pbpl import common
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
import string

#rodan:~/pbpl-compton/share/baby-cpt/analysis/nonstripe4500/MLE> 
#python MLE.py --h5_name pwfa5e10-log-cropped --dir ../fake-data --config ../config/nonstripe4500-Emily-Lisa-Rose-Adam-Bella-THABIT.toml --out jan29-original

#python MLE.py --out April25_MLE --lower_cutoff 2e-1 --h5_name pwfa5e10-log-cropped --dir ../fake-data --config ../config/nonstripe4500-Emily-Lisa-Rose-Adam-Bella-THABIT.toml 

#python MLE.py --out May2_MLE_original --lower_cutoff 2e-1 --h5_name pwfa5e10-log-cropped --dir ../fake-data --config ../config/nonstripe4500-Emily-Lisa-Rose-Adam-Bella-THABIT.toml 


def get_weights_gaussian(num_simulations, sigma):
    mu = random.randint(int(0-sigma*2), int(num_simulations+sigma*2))
    #variance = 60
    #sigma = 40
    x = np.linspace(0, num_simulations, num_simulations)
    float_weights = stats.norm.pdf(x, mu, sigma)
    return float_weights

def get_gaussian(length, mu, sigma):
    #mu = random.randint(int(0-sigma*2), int(num_simulations+sigma*2))
    #variance = 60
    #sigma = 40
    x = np.arange(length)
    float_weights = stats.norm.pdf(x, mu, sigma)
    return float_weights

def get_basis(edep, num_events, energy_ranges, x_max, 
    x_bins, y_max, y_bins, 
    l_y_lower, l_y_upper, l_y_bins,
    l_e_lower, l_e_upper, l_e_bins):

    num_events = np.squeeze(np.array(num_events))#100
    print("num_events.shape", num_events.shape)
    energy_ranges = np.squeeze(np.array(energy_ranges))#101
    print('energy_ranges', energy_ranges)
    print("energy_ranges.shape", energy_ranges.shape)
    energy_range_lengths = np.diff(energy_ranges)#100
    energy_midpoints = energy_ranges[0:-1]+energy_range_lengths/2

    resampled_energy_ranges = np.logspace(np.log(l_e_lower), np.log(l_e_upper), l_e_bins+1, base=np.e)

    edep = np.array(edep).sum(axis=1)
    edep_resized = resize(edep, (len(edep), y_bins, x_bins))

    plt.imshow(edep_resized[len(edep)//2])
    plt.savefig("test_edep_resized.png")

    edep_transposed = edep_resized.transpose(1, 2, 0)

    # the avaraged response of one MeV in the energy range (a, b)
    edep_transposed_averaged = edep_transposed/(num_events*energy_midpoints) #correct
    print(edep_transposed_averaged[0][1][1])

    train_data = np.empty([1, x_bins])
    train_labels = np.array([np.linspace(0, 31, l_e_bins*l_y_bins)])

    num_simulations = len(num_events)

    for i in range(0, l_e_bins):
        
        e_float_weights = np.zeros([l_e_bins])
        e_float_weights[i] = 1

        e_weights = e_float_weights#/np.max(e_float_weights)#.astype(int)
        #want this to be energy_density. 
        #amount of energy between i MeV, j MeV is energy_density((i+j)/2)*(j-i)
        #print(e_weights)
        weights = np.interp(energy_ranges[:-1], resampled_energy_ranges[:-1], e_weights)#
    
        ###################### key step ######################
        new_edep = np.dot(edep_transposed_averaged, weights*energy_range_lengths)
        ######################################################
        
        histo = np.array([e_weights]) #make sure that histo is uniformly random
        #print(histo)
        #return
        #ye = np.column_stack((ys, es))
        
        train_data = np.append(train_data, [new_edep[y_bins//2]], axis=0)
        train_labels = np.append(train_labels, [histo.flatten()], axis = 0)
        #print("train_labels.shape", train_labels.shape)

    print("train_data.shape", train_data.shape)
    train_data = train_data[1:].astype(float)
    train_labels = train_labels[1:].astype(float)

    return train_labels, train_data


# There are 100 gamma energy bins in the training data.  
# Though the bin energies are logarithmically spaced, 
# the gammas are sampled linearly within each bin.  

def get_gaussian_basis(edep, num_events, energy_ranges, x_max, 
    x_bins, y_max, y_bins, 
    l_y_lower, l_y_upper, l_y_bins,
    l_e_lower, l_e_upper, l_e_bins):

    num_events = np.squeeze(np.array(num_events))#100
    print("num_events.shape", num_events.shape)
    energy_ranges = np.squeeze(np.array(energy_ranges))#101
    print("energy_ranges.shape", energy_ranges.shape)
    energy_range_lengths = np.diff(energy_ranges)#100
    energy_midpoints = energy_ranges[0:-1]+energy_range_lengths/2

    resampled_energy_ranges = np.logspace(np.log(l_e_lower), np.log(l_e_upper), l_e_bins+1, base=np.e)

    edep = np.array(edep).sum(axis=1)
    edep_resized = resize(edep, (len(edep), y_bins, x_bins))
    edep_transposed = edep_resized.transpose(1, 2, 0)

    # the avaraged response of one MeV in the energy range (a, b)
    edep_transposed_averaged = edep_transposed/(num_events*energy_midpoints) #correct
    print(edep_transposed_averaged[0][1][1])

    train_data = np.empty([1, x_bins])
    train_labels = np.array([np.linspace(0, 31, l_e_bins*l_y_bins)])

    num_simulations = len(num_events)

    for i in range(0, l_e_bins):
        
        e_component_weights = np.stack((get_gaussian(l_e_bins, i, l_e_bins//20)))
        function_weights = 1
        e_float_weights = ((np.dot(e_component_weights.T, function_weights)).T)
        e_weights = e_float_weights#/np.max(e_float_weights)#.astype(int)
        #want this to be energy_density. 
        #amount of energy between i MeV, j MeV is energy_density((i+j)/2)*(j-i)
        #print(e_weights)
        weights = np.interp(energy_ranges[:-1], resampled_energy_ranges[:-1], e_weights)#
    
        ###################### key step ######################
        new_edep = np.dot(edep_transposed_averaged, weights*energy_range_lengths)
        ######################################################
        
        histo = np.array([e_weights]) #make sure that histo is uniformly random
        #print(histo)
        #return
        #ye = np.column_stack((ys, es))
        
        train_data = np.append(train_data, [new_edep[y_bins//2]], axis=0)
        train_labels = np.append(train_labels, [histo.flatten()], axis = 0)
        #print("train_labels.shape", train_labels.shape)

    print("train_data.shape", train_data.shape)
    train_data = train_data[1:].astype(float)
    train_labels = train_labels[1:].astype(float)

    return train_labels, train_data

def error_square(guess, yi, pi):
    #yi: [num, l_y_bins, l_e_bins]
    yi = np.reshape(yi, [yi.shape[0], -1])
    pi = np.reshape(pi, [pi.shape[0], -1])
    #guess = (np.random.rand(l_y_bins*l_e_bins, x_bins*y_bins), np.random.rand(l_y_bins*l_e_bins))
    guess = np.reshape(guess, [yi.shape[1], pi.shape[1]+1])
    thetaT = guess [:yi.shape[1], :pi.shape[1]]
    intercept = guess [yi.shape[1]-1:, pi.shape[1]-1:]
    yhat = np.dot(thetaT, pi.transpose(1, 0))+intercept
    error = np.sum((yi.transpose(1, 0)-yhat)**2)
    return error

def iterate_shepp_vardi(x0, R, y):
    y0 = np.matmul(x0, R)
    mask = (y0 != 0)
    yrat = np.zeros_like(y)
    yrat[mask] = y[mask]/y0[mask]
    return (x0/R.sum(axis=1))*np.matmul(R, yrat)

def main(args):

    conf = toml.load(args.config)
    #f["compton-electron']['edep']<HDF5 dataset "edep": shape (100, 1, 5, 300, 450), type "<f4">
    gin = h5py.File('../fake-data/4500.h5')
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

    histo, R = get_gaussian_basis(edeps, num_events, energy_ranges, x_max, #get_gaussian_basis(edeps, num_events, energy_ranges, x_max, 
    x_bins, y_max, y_bins, 
    l_y_lower, l_y_upper, l_y_bins,
    l_e_lower, l_e_upper, l_e_bins)

    dirname = args.dir 
    name = args.h5_name
    filename = dirname+'/'+name+'.h5'
    #
    #filename = 'test-right-y/varY-col-2e6-UPFZET-0.h5'

    gin = h5py.File(filename, 'r')
    edep = gin['compton-electron']['edep'][0] 
    edep = edep.sum(axis=0)
    edep_resized = resize(edep, (y_bins, x_bins))

    edep_normalized = edep_resized
    y = edep_normalized[int(y_bins/2)]
    try:
        with np.load(dirname+'/'+name+'.npz') as data:
            truth = data['histo']
    except (IOError):
        truth = None

    #minimize least square error
    #guess = np.random.rand(l_y_bins*l_e_bins*(x_bins+1))
    #print(guess.shape)
    #results = minimize(error_square, guess, args=(yi, pi), method = "Nelder-Mead", options={'disp': True})

    #print(results.x)
    #print(results.success)
    x0 = np.ones([l_e_bins])*1e7
    x1 = iterate_shepp_vardi(x0, R, y)
    records=[]
    records.append(x1)
    count = 0
    while (count<50):   #np.sum(x0-x1)>1e-5 and
        print ("step:", count)
        x0=x1
        x1=iterate_shepp_vardi(x0, R, y)
        binz = np.arange(len(x1))
        #x1 *= np.tanh(binz/10)
        count += 1
        records.append(x1)

    test_predictions = np.dot(x1, histo)
    print('test_predictions.shape', test_predictions.shape)

    common.setup_plot()
    #fig3 = plt.figure(figsize=(3+3/8, 4+4/8), dpi=600, constrained_layout=True)
    #gs = fig3.add_gridspec(nrows=2, ncols=1, height_ratios = [10., 7.])
    fig3 = plt.figure(figsize=(3+3/8, 2+2/8), dpi=600, constrained_layout=True)
    gs = fig3.add_gridspec(nrows=1, ncols=1)

    f3_ax1 = fig3.add_subplot(gs[0, 0])
    photon_e_bins = np.logspace(np.log(l_e_lower), np.log(l_e_upper), l_e_bins+1, base=np.e)
    print(photon_e_bins)
        
        #f3_ax1_c = fig3.add_subplot(gs[0, 3])
        #fig3.colorbar(truth_im, cax=f3_ax1_c, shrink=0.6)
    records = np.array(records)
    records = records.reshape((-1, l_y_bins, l_e_bins))
    prediction = test_predictions.reshape((l_y_bins, l_e_bins))
    print('prediction', prediction)
    prediction_c = prediction[int(l_y_bins/2)]
    records_c = records[:, l_y_bins//2]
    print('prediction_c', prediction_c)
    f3_ax1.plot(photon_e_bins[:-1], prediction_c.clip(min=0), label = "prediction")#color='r', 
    #for i in range(len(records)):
    #    c = [float(i)/float(len(records)), 0.0, float(len(records)-i)/float(len(records))] #R,G,B
    #    f3_ax1.plot(photon_e_bins[:-1], records_c[i], color=c)
    if truth is not None:
        truth_c = truth[int(l_y_bins/2)]
        f3_ax1.plot(photon_e_bins[:-1], truth_c, label = "truth")#color='g', 
        #f3_ax1.set_title("truth vs prediction")
        #RMSE Calculation
        #normalize
        max_truth = np.max(truth_c)
        truth_c_normalized = truth_c/max_truth
        prediction_normalized = prediction_c.clip(min=0)/max_truth
        RMSE = np.sqrt(np.mean((prediction_normalized-truth_c_normalized)**2))
        print(f"---------------max truth: {max_truth}, RMSE: {RMSE}---------------")
        #gaussian ---------------max truth: 5594892069.26858, RMSE: 0.03962025357639625---------------
        #original ---------------max truth: 5594892069.26858, RMSE: 0.2056961019119214---------------


    #pre_im = f3_ax2.imshow(prediction)
    #f3_ax2.set_title("prediction")
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
    f3_ax3.plot(np.linspace(0, x_max, x_bins+1)[:-1], edep_normalized[int(y_bins/2)])
    f3_ax3.set_title("energy deposition")
    #f3_ax3_c = fig3.add_subplot(gs[:, 1])
    #fig3.colorbar(im, cax=f3_ax3_c, shrink=0.6)
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(truth_im, cax=cbar_ax)
    '''

    plt.savefig(args.out+"-"+name+".png")
    np.savez(args.out+"-"+name+".npz", prediction=prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict gamma spectra based on spectrometer data.')
    parser.add_argument("--h5_name", required=True,
        help="set where spectrometer output is.")
    parser.add_argument("--dir", required=True,
        help="set where the h5 file is in")
    parser.add_argument("--config", required=True,
        help="the config toml file")
    parser.add_argument("--out", required=True,
        help="output file name suffix")
    parser.add_argument("--lower_cutoff", type=float,
        help="set the lower bound for the plot.")

    print("THIS IS THE UPDATED VERSION")
    
    args = parser.parse_args()
    main(args)

