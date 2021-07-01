#!/usr/bin/env python
import sys
import os
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import simps
from scipy.stats import rv_histogram
from scipy.spatial import cKDTree
from scipy.interpolate import griddata, interp1d, RectBivariateSpline, interp2d

import time


def generate_3d_distrib(xin, yin, zin, pdf3d, num, eps=2.2e-16):
    samples = np.zeros([num, 3])
    #https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    combined_x_y_arrays = np.transpose([np.tile(xin, len(yin)), np.repeat(yin, len(xin))])
    print(combined_x_y_arrays)
    mytree = cKDTree(combined_x_y_arrays)

    xpdf = pdf3d.sum(axis=(1,2))
    print('xpdf')
    print(xpdf)
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
            print(i)
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
                print(repr(zpdf))
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
    fin = h5py.File('0099.h5', 'r')
    px = np.array(fin["px"][0])
    py = np.array(fin["py"][0])
    pz = np.array(fin["pz"][0])
    energy = np.sqrt(px**2+py**2+pz**2)*3e8/1.602e-18
    print(max(energy), min(energy))
    w_p = (fin["w_p"][0])
    y, x = np.histogram(energy*w_p, bins=50)
    x = x[:-1]
    fig = plt.figure()
    ax = fig.add_subplot()
    mask = y>0
    ax.scatter(x[mask],y[mask])
    ax.set_yscale('log')
    ax.set_ylabel("particle weight")
    ax.set_xlabel("energy ??")
    plt.show()
    # px py pz w_p
    # xxx yyy zzz n
    
    sample_energy = [1, 2, 3, 4, 5]
    weights = [1, 1, 1, 2, 3]
    photons_w_energy =[1, 2, 3, 8, 15]
    y, x = np.histogram(sample, weights=weights)
    x = x[:-1]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_ylabel("particle weight")
    ax.set_xlabel("energy ??")
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
