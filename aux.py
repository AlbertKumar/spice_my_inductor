import numpy as np
from matplotlib import pyplot as plt
from skrf_extensions import R, L
from pathlib import Path
from itertools import product
import time
import inspect

"""
Auxiliary functions used to plot and generate PNGs.
"""


def bffit(parameters=None, obj_fcn=None, args=None, steps=10, logstep=False, verbose=False):
    """
    Performs a brue force fit.
    Requires Python 3.6+ for ordered dictionary.
    :param parameters: <lmfit Parameters>
    :param obj_fcn: User defined objective function to minimize. <Python function>
    :param args: A tuple or list of arguments to the objective function. <tuple or list>
    :param steps: Number of brute force steps for each of the parameters. <int>
    :param logstep: False=linearly spaced steps. True=logarithmically spaced steps. <Bool>
    :param verbose: Provides some text outputs. <Bool>
    :return:
    """
    # Check if the obj_fcn is a method of a class. If it is, 'self' need to be the first argument.
    #obj_fcn_args = inspect.getfullargspec(obj_fcn)
    #print("inspect",obj_fcn_args[0])

    # Determine each parameters min and max.
    param_ranges = {}
    for param in parameters:
        if logstep:
            param_ranges[param] = np.geomspace(parameters[param].min, parameters[param].max, steps)
        else:
            param_ranges[param] = np.linspace(parameters[param].min, parameters[param].max, steps)

    # Loop through all combinations the parameters.
    combo_error = {}
    combo_best = None
    error_prev = 1e9
    for combo in product(*param_ranges.values()):
        keys = list(parameters.valuesdict().keys())
        params_tmp = {}
        for key, val in zip(keys, combo):
            params_tmp[key] = val
        error = obj_fcn(params_tmp, *args)
        combo_error[combo] = error

        # Save the best combo.
        if error < error_prev:
            combo_best = {key: val for (key, val) in zip(keys, combo)}
            #print("found", combo_best)
            error_prev = error
    if verbose:
        print("1st optimization:", combo_best)

    # Get the top 3 combos, but make sure the values are not repeated for any single parameter.
    # Find a combo with a higher and lower value for each parameter.
    # best_n_combos = list(combo_error_sorted.keys())[:3]
    combo_error_sorted = dict(sorted(combo_error.items(), key=lambda item: item[1]))
    best_combo = list(combo_error_sorted.keys())[0]
    best_3_combos = []
    parameters_best_val = {k: v for k, v in zip(parameters, best_combo)}
    parameters_low_val = {k: None for k in parameters}
    parameters_high_val = {k: None for k in parameters}

    for combo in combo_error_sorted:
        for param, val in zip(parameters, combo):
            if val < parameters_best_val[param] and parameters_low_val[param] is None:
                if val < parameters[param].min:
                    val = parameters[param].min
                parameters_low_val[param] = val
            elif val > parameters_best_val[param] and parameters_high_val[param] is None:
                if val > parameters[param].max:
                    val = parameters[param].max
                parameters_high_val[param] = val

        # Break out of loop if all the values are found (i.e. no None in the parameters_*_val dicts).
        if all(parameters_low_val.values()) and all(parameters_high_val.values()):
            break

    # Handle the case where the best_val is the high or low.
    # For example, if the parameter doesn't vary or the best point is the lowest point,
    # then a None will remain in parameters_high_val or parameters_low_val dicts.
    if not all(parameters_high_val.values()):
        for param in parameters_high_val:
            if parameters_high_val[param] is None:
                parameters_high_val[param] = parameters[param].max
    if not all(parameters_low_val.values()):
        for param in parameters_low_val:
            if parameters_low_val[param] is None:
                parameters_low_val[param] = parameters[param].min

    # Repeat the optimization with the new ranges (fine tune).
    param_ranges = {}
    for param in parameters:
        if logstep:
            param_ranges[param] = np.geomspace(parameters_low_val[param], parameters_high_val[param], steps)
        else:
            param_ranges[param] = np.linspace(parameters_low_val[param], parameters_high_val[param], steps)

    combo_error = {}
    combo_best = None
    error_prev = 1e9
    for combo in product(*param_ranges.values()):
        keys = list(parameters.valuesdict().keys())
        params_tmp = {}
        for key, val in zip(keys, combo):
            params_tmp[key] = val
        error = obj_fcn(params_tmp, *args)
        combo_error[combo] = error
        if error < error_prev:
            combo_best = {key: val for (key, val) in zip(keys, combo)}
            error_prev = error
    if verbose:
        print("2nd optimization:", combo_best)

    # Write out results.
    parameters_best_val = {k: v for k, v in zip(parameters, best_combo)}

    return parameters_best_val


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

