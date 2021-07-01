#!/usr/bin/env python
import sys
import os
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import numpy as np
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import simps
from scipy.stats import rv_histogram
from scipy.spatial import cKDTree
from scipy.interpolate import griddata, interp1d, RectBivariateSpline, interp2d
from pbpl import common
from pbpl.common.units import *
import time

mrad = 1e-3
steradian = 1.0
def log_minor_ticks(axis):
    locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12)
    axis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(
        base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
    axis.set_minor_locator(locmin)
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())

def plot_spectral_photon_density(
        output, fin, group_names, xscale, yscale, xlim, ylim,
        labels=None, display_legend=True):
    fig = plot.figure(figsize=(244.0/72, 140.0/72))
    ax = fig.add_subplot(1, 1, 1)

    plot.xlabel(r'Photon energy ({})'.format(xscale[1]), labelpad=0.0)
    plot.ylabel(
        r'Spectral photon density ({})'.format(yscale[1]), labelpad=1.0)

    for i, group_name in enumerate(group_names):
        g = fin[group_name]
        photon_energy = g['energy'][:]*MeV
        thetax = g['thetax'][:]*mrad
        thetay = g['thetay'][:]*mrad
        print("thetax")
        print(thetax.shape)#201

        d2W = g['d2W'][:]*joule/(mrad**2*MeV)
        print("d2W")
        print(d2W.shape)#100, 201, 201
        dthetax = thetax[1]-thetax[0]
        dthetay = thetay[1]-thetay[0]

        spectral_energy_density = d2W.sum(axis=(1,2))*dthetax*dthetay
        spectral_photon_density = spectral_energy_density/photon_energy
        xq = np.array(photon_energy)

        print("photon energy")
        #print(xq)
        print("spectral photon density")
        pdf = np.array(spectral_photon_density)
        #print(repr(pdf))
        print("_________________")

        label = group_name
        if labels != None:
            label = labels[i]

        mask = spectral_photon_density>0
        ax.semilogy(
            np.concatenate(((0.0,), photon_energy[mask]/xscale[0])),
            np.concatenate(((0.0,), spectral_photon_density[mask]/yscale[0])),
            linewidth=0.6,
            label=label)

        num_photons = simps(spectral_photon_density, photon_energy)
        data=generate_distrib(photon_energy/xscale[0], spectral_photon_density/yscale[0], 1000)
        #print(data)
        #ax.hist(data, bins=100, label=label, range = [-0.1, 2.0])

    if display_legend:
        ax.legend()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    log_minor_ticks(ax.yaxis)

    output.savefig(fig, transparent=True)

def plot_d2W(output, fin, group_name, label = None):
    g = fin[group_name]
    photon_energy = g['energy'][:]*MeV
    thetax = g['thetax'][:]*mrad
    thetay = g['thetay'][:]*mrad
    d2W = g['d2W'][:]*joule/(mrad**2*MeV) #e, x, y
    dthetax = thetax[1]-thetax[0]
    dthetay = thetay[1]-thetay[0]

    print("d2W x, y")
    #print(d2W[:, 100, 100])

    fig = plot.figure(figsize=(244.0/72, 140.0/72))
    ax = fig.add_subplot(1, 1, 1)
    d2W_2d = d2W.sum(axis=1)
    ax.imshow(np.transpose(d2W_2d)/photon_energy)
    output.savefig(fig, transparent=True)

    print("photon_energy")
    #print(repr(photon_energy))

    xye = np.transpose(d2W, axes=(1, 2, 0))/photon_energy
    print("d2W x, y")
    #print(xye[100, 100, :])
    samples = generate_3d_distrib(thetax, thetay, photon_energy, xye, 2000)
    name="10000-points-test"
    np.savez("{}.npz".format(name), points=samples)
    name="3d-array-sfqed"
    np.savez("{}.npz".format(name), 
        theta_x=theta_x, theta_y=theta_y, energy=photon_energy, array=xye)
    #with np.load(name+'.npz') as data:
        #samples = data['points']
        #print(samples.shape)
    fig = plot.figure(figsize=(244.0/72, 140.0/72))
    ax = fig.add_subplot(1, 1, 1)
    print("sample energy")
    #print(samples[:, 2])
    ax.hist2d(samples[:, 1], samples[:, 2], bins=[100, 100],
        range=[[-max(thetay), max(thetay)], [0, max(photon_energy)]])
    output.savefig(fig, transparent=True)

def plot_double_differential(
        output, fin, group_name, num_contours,
        xscale, yscale, zscale, xlim, ylim, label=None):
    g = fin[group_name]
    photon_energy = g['energy'][:]*MeV
    thetax = g['thetax'][:]*mrad
    thetay = g['thetay'][:]*mrad
    d2W = g['d2W'][:]*joule/(mrad**2*MeV)
    dthetax = thetax[1]-thetax[0]
    dthetay = thetay[1]-thetay[0]

    fig = plot.figure(figsize=(244.0/72, 140.0/72))
    ax = fig.add_subplot(1, 1, 1)

    plot.xlabel(r'Photon energy ({})'.format(xscale[1]), labelpad=0.0)
    plot.ylabel(r'$\theta_x$ ({})'.format(yscale[1]), labelpad=0.0)

    x = np.linspace(*xlim, 100)
    y = np.linspace(*ylim, 100)
    post_shape = np.array(list(itertools.product(x, y)))

    foo = np.array(list(itertools.product(
        photon_energy/xscale[0], thetax/yscale[0])))
    bar = simps(d2W, thetay, axis=2).flatten()
    A = griddata(foo, bar, post_shape, method='cubic')
    mask = (A < A.max()*0.001)
    A[mask] = A.max()*0.001

    contours = ax.contourf(
        x, y, A.reshape((100,100)).T / zscale[0],
        levels=num_contours, cmap=common.blue_cmap, vmin=0)
    ax.contour(
        x, y, A.reshape((100,100)).T / zscale[0],
        levels=num_contours, colors='k', linewidths=0.4, vmin=0)

    cbar = plot.colorbar(contours)
    cbar.ax.set_ylabel(zscale[1])

    if label == None:
        label = group_name
    ax.text(0.02, 0.9, label, transform=ax.transAxes, size=7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    output.savefig(fig, transparent=True)

def generate_3d_distrib(xin, yin, zin, pdf3d, num, eps=2.2e-16):
    samples = np.zeros([num, 3])
    #https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    combined_x_y_arrays = np.transpose([np.tile(xin, len(yin)), np.repeat(yin, len(xin))])
    print(combined_x_y_arrays.shape())
    mytree = cKDTree(combined_x_y_arrays)

    xpdf = pdf3d.sum(axis=(1,2))
    print('xpdf')
    print(xpdf.shape())
    xbins = np.append(xin, xin[-1]+xin[1]-xin[0])
    ybins = np.append(yin, yin[-1]+yin[1]-yin[0])
    zbins = np.append(zin, zin[-1]+zin[1]-zin[0])
    rv = rv_histogram((xpdf, xbins))
    xsamples = rv.rvs(size=num)
    samples[:, 0] = xsamples
    pdf2d = pdf3d.sum(axis=2)
    yfunc = interp1d(xin, pdf2d, axis=0)
    #zfunc = RectBivariateSplineAxis12(xin, yin, pdf3d, axis=1)
    ypdfs = yfunc(xsamples)

    for i in range(num):
        rv = rv_histogram((ypdfs[i], ybins))
        y = rv.rvs()
        samples[i, 1] = y
        if(i%1000==0):
            pass#print(i)
        x = xsamples[i]
        dist, index = mytree.query([x, y])
        #print(dist)
        if(True):
            #print("use kdt tree")
            #xn, yn = combined_x_y_arrays[index]
            #print(xn, yn)
            xindex = index % len(xin)
            yindex = (index - xindex)/len(xin)
            #print(yindex)
            zpdf = pdf3d[xindex, int(yindex), :]
            if(i%1000==0):
                print("zpdf")
                #print(repr(zpdf))
            rv = rv_histogram((zpdf, zbins))
        else:
            zpdf = RectBivariateSplineAxis12(xin, yin, pdf3d, xsamples[i], y)
            rv = rv_histogram((zpdf, zbins))
        samples[i, 2] = rv.rvs()
    return samples

def RectBivariateSplineAxis12(x, y, z, xi, yi):
    values = np.zeros(z.shape[2])
    for i in range(len(values)):
        #RBS = interp2d(x, y, z[:, :, i])
        #values[i] = RBS(xi, yi)

        RBS = RectBivariateSpline(x, y, z[:, :, i], kx=1, ky=1)
        values[i] = RBS.ev(xi, yi)
    return values

def nearestGridPoint(x, y, xin, yin):
    #https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
    # Shoe-horn existing data for entry into KDTree routines
    print("run kdt tree")
    combined_x_y_arrays = np.dstack([yin.ravel(),xin.ravel()])[0]
    point = [x, y]
    mytree = cKDTree(combined_x_y_arrays)
    dist, index = mytree.query(point)
    return combined_x_y_arrays[index]

def generate_distrib(x, pdf, num):
    x_bins = np.append(x, x[-1]+1*MeV)
    x_binwidth = (x_bins[1:] - x_bins[:-1])
    rv = rv_histogram((pdf, x_bins))
    samples = rv.rvs(size=num)
    return samples

def main():
    fin = h5py.File('d2W.h5', 'r')
    common.setup_plot()
    os.makedirs('figs', exist_ok=True)

    # PWFA
    if True:
        sfqed_datasets = [
            'SFQED/MPIK/LCFA_w3.0_xi5.7',
            'SFQED/MPIK/LCFA_w2.4_xi7.2',
            'SFQED/MPIK/LCS+LCFA_w3.0_xi5.7',
            'SFQED/MPIK/LCS+LCFA_w2.4_xi7.2']
        sfqed_labels = [
            r'LCFA ($a_0=5.7, w_0=3.0\;\mu{\rm m}$)',
            r'LCFA ($a_0=7.2, w_0=2.4\;\mu{\rm m}$)',
            r'QED ($a_0=5.7, w_0=3.0\;\mu{\rm m}$)',
            r'QED ($a_0=7.2, w_0=2.4\;\mu{\rm m}$)']
        output = PdfPages('figs/sfqed-test.pdf')
        fig = plot.figure(figsize=(244.0/72, 140.0/72))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.linspace(0, 10, 10), np.linspace(0, 20, 10))
        output.savefig(fig, transparent=True)
        #plot_spectral_photon_density(
        #    output, fin, pwfa_trailing_datasets[1:2],
        #    (MeV, 'MeV'),
        #    (1/MeV, '1/MeV'),
        #    (-0.1, 2.0), (1e7, 1e13))
        plot_spectral_photon_density(
            output, fin, sfqed_datasets,
            (GeV, 'GeV'), (1/GeV, '1/GeV'),
            (-0.1, 10.0), (1e1, 1e9),
            labels=sfqed_labels)

        #plot_d2W(
        #    output, fin, pwfa_trailing_datasets[1],
        #    )

        #for x in pwfa_trailing_datasets:
        #    plot_double_differential(
        #        output, fin, x, 8,
        #        (MeV, 'MeV'), (mrad, 'mrad'),
        #        (uJ/(mrad*MeV), r'uJ/(mrad$^\cdot$MeV)'),
        #        (0.01, 1.0), (-2.0, 2.0))
        #output.close()

if __name__ == '__main__':
    sys.exit(main())
