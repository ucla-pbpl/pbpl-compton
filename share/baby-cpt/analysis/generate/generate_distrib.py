import scipy.stats as stats

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

def generate_distrib(lower, upper, stats_func, num, param=[]):
    nrm=stats_func.cdf(upper, *param)-stats_func.cdf(lower, *param)
    xrng=np.arange(lower,upper,(upper-lower)/50)
    yr=rnd.rand(num)*(nrm)+stats_func.cdf(lower, *param)
    xr=stats_func.ppf(yr, *param)
    return xr

def main():
    #plot the original distribution
    xrng=np.arange(-10,10,.1)
    yrng=stats.logistic.pdf(xrng)
    plt.plot(xrng,yrng)

    #plot the truncated distribution
    nrm=stats.logistic.cdf(5)-stats.logistic.cdf(0)
    xrng=np.arange(0,5,.01)
    yrng=stats.logistic.pdf(xrng)/nrm
    plt.plot(xrng,yrng)

    #sample using the inverse cdf
    yr=rnd.rand(100000)*(nrm)+stats.logistic.cdf(0)
    xr=stats.logistic.ppf(yr)
    plt.hist(xr,density=True)

    #plot the original distribution
    a = 3
    xrng=np.arange(-10,10,.1)
    yrng=stats.dgamma.pdf(xrng, a)
    plt.plot(xrng,yrng)

    xr = generate_distrib(-5, 5, stats.dgamma, 100000, [a])
    plt.hist(xr,density=True)

    xr = generate_distrib(0, 1, stats.rayleigh, 100000)
    plt.hist(xr,density=True)

    fig3 = plt.figure(constrained_layout=True)
    gs = fig3.add_gridspec(2, 2)
    f3_ax1 = fig3.add_subplot(gs[0, 1])
    f3_ax1.set_title('gs[0, 1]')
    f3_ax1 = fig3.add_subplot(gs[1, 1])
    f3_ax1.set_title('gs[1, 1]')
    f3_ax2 = fig3.add_subplot(gs[:, 0])
    f3_ax2.set_title('gs[:, 0]')

    plt.show()

if __name__ == '__main__':
    main()
