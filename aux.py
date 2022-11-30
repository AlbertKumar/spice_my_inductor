import numpy as np
from matplotlib import pyplot as plt
from skrf_extensions import R, L
from pathlib import Path

"""
Auxiliary functions used to plot and generate PNGs.
"""

def plotLR(ntwk1, ntwk2=None, plot_type='series', filename=None):
    f = ntwk1.frequency.f
    if plot_type == "series":
        p1 = 0
        p2 = 1
    elif plot_type == "in":
        p1 = 0
        p2 = 0
    R1 = R(ntwk1, p1, p2)
    L1 = L(ntwk1, p1, p2)
    Q1 = -ntwk1.y[:, p1, p2].imag / ntwk1.y[:, p1, p2].real

    if ntwk2 is not None:
        R2 = R(ntwk2, p1, p2)
        L2 = L(ntwk2, p1, p2)
        Q2 = -ntwk2.y[:, p1, p2].imag / ntwk2.y[:, p1, p2].real

    fig, ax = plt.subplots(3, sharex=True)
    fig.canvas.set_window_title('{} fit'.format(plot_type))
    ax[0].plot(f, R1, linestyle='None', marker='o', markeredgecolor='b', markersize=5, alpha=0.1)
    if ntwk2 is not None: ax[0].plot(f, R2, linestyle='--', color='k')
    ax[0].set(ylabel='R{} (ohms)'.format(plot_type))
    #ax[0].set_ylim(0.9*min(R1), 1.1*max(R1))
    ax[0].set_yscale('log')

    ax[1].plot(f, L1, linestyle='None', marker='o', markeredgecolor='b', markersize=5, alpha=0.1)
    if ntwk2 is not None: ax[1].plot(f, L2, linestyle='--', color='k')
    ax[1].set(ylabel='L{} (nH)'.format(plot_type))
    ax[1].set_ylim(0.9*min(L1), 1.3*max(L1))

    ax[2].plot(f, Q1, linestyle='None', marker='o', markeredgecolor='b', markersize=5, alpha=0.1)
    if ntwk2 is not None: ax[2].plot(f, Q2, linestyle='--', color='k')
    ax[2].set(ylabel='Q{}'.format(plot_type))
    ax[2].set_ylim(0, max(Q1))
    ax[2].set(xlabel='frequency (Hz)')
    ax[2].set_xscale('log')

    # Save figure.
    if filename is not None:
        path = Path(filename)
        plt.savefig(path)


def quickplot(x, y, y2=None, yrange=None, ystep=None, logx=False, logy=False):
    fig, ax = plt.subplots()
    ax.plot(x, y, linestyle='None', marker='o', markeredgecolor='b', markersize=3, alpha=0.25)
    if y2 is not None: ax.plot(x, y2, linestyle='--')
    if yrange:
        ax.set_ylim(yrange[0], yrange[1])
        if ystep: ax.set_yticks(np.arange(yrange[0], yrange[1]+ystep, ystep))
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')


def get_idx_at(value, array):
    '''
    Returns index of array element with the closest value.
    :param value: value to match
    :param array: array
    :return: Index of array element with the closest value.
    '''
    idx = (np.abs(array - value)).argmin()
    return idx

