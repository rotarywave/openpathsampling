"""This module defined to collect function needed for interface positions optimization.

Important methods defined here
------------------------------

find_interface_and_cross
    Reads the output of the analyze-tool after a certain number of cycles
    (defined by the user). Information about 'transition' and 'histograms' must be provided.

get_interpolation_points:
    A method to calculate the points of the transformation, that is going to map required
    sequence to the new values of the interface positions.

new_interface_A
    A method for calculation of the new interface positions, using I approach: the number of interfaces
    kept the same, only the values of the positions are going to be changed.

new_interface_B
    A method for calculation of the new interface positions, using II approach: the number of
    the interfaces might be changed, "expected" crossing probability for each ensemble has a
    predefined value.

"""
import matplotlib.pyplot as plt
import numpy as np
import openpathsampling as paths
from scipy.interpolate import interp1d
from openpathsampling.numerics import SparseHistogram

def find_interface_and_cross(transition):  # pragma: no cover
    """Read the output of the analyze tool, and get interface positions and crossing probabilities.
    Example for getting input:
    transition = system.transitions[(stateA, stateB)]
    histo = transition.histograms['max_lambda']

    Parameters
    ----------
    transition : an object
        Information about transition

    Returns
    -------
    old_interfaces : list of floats
        Positions of the interfaces in the simulation
    old_crossing_probabilities : list of floats
        Crossing probabilities for each ensemble
    """
    histo = transition.histograms['max_lambda']
    old_interfaces = transition.interfaces.lambdas
    all_probabilities = []
    for item in histo:
        all_probabilities.append(histo[item].reverse_cumulative())

    old_crossing_probabilities = []
    for item in old_interfaces:
        old_crossing_probabilities.append(all_probabilities[1](item))
    return old_interfaces, old_crossing_probabilities


def get_interpolation_points(pos_interfaces, cross_probabilities):
    """Calculated the point for transformation funstion.

    Parameters
    ----------
    pos_interfaces : list of floats
        Old interface positions
    cross_probabilities : list of floats
        Old crossing probabilities for each ensemble

    Returns
    -------
    out[0] : list of floats
        x-values for interpolation
    out[1] : list of floats
        y-values for interpolation
    """
    
    logarithms = np.log(cross_probabilities)
    point_values = []
    for i in range(len(pos_interfaces)):
        point_values.append(sum(logarithms[:(i+1)])/float(sum(logarithms)))
    # now create a set of base points for interpolation function
    x = point_values[:]
    y = pos_interfaces[:]
    return x, y
        

def new_interfaces_A(interfaces, cross_probabilities):
    """Calculates the new interfaces, if the number of the interfaces remains the same.

    Parameters
    ----------
    interfaces : list of floats
        Old interface positions
    cross_probabilities : list of floats
        Old crossing probabilities for each ensemble

    Returns
    -------
    new_interfaces : list of floats
        New positions of the interfaces
    """
    
    x_flambda, y_lambda = get_interpolation_points(interfaces, cross_probabilities)
    # build transformation function
    f = interp1d(x_flambda, y_lambda, kind='linear')
    # define the number of interfaces
    if len(interfaces) < 3:
        N = 3
    else:
        N = len(interfaces)

    # find the interfaces
    f_interval = ( x_flambda[-1] - x_flambda[0] ) / float(N-1)
    new_interfaces = []
    for i in range(N):
        value = f(x_flambda[0] + f_interval * i)
        new_interfaces.append(float(value))
    return new_interfaces


def new_interfaces_B(interfaces, cross_probabilities, p=0.5):
    """Calculates the new interfaces, if the crossing probabilities expected to have some fixed value.

    Parameters
    ----------
    interfaces : list of floats
        Old interface positions
    cross_probabilities : list of floats
        Old crossing probabilities for each ensemble
    p : float
        Expected value for crossing probabilities in each ensemble. Default 0.5.

    Returns
    -------
    new_interfaces : list of floats
        New positions of the interfaces
    """

    x_flambda, y_lambda = get_interpolation_points(interfaces, cross_probabilities)
    # build interpolation function
    f = interp1d(x_flambda, y_lambda, kind='linear')
    # find the number of interfaces
    total_cross_prob = 1
    for item in cross_probabilities:
        total_cross_prob = total_cross_prob * item
    if (int(np.log(total_cross_prob) / (np.log(p)) + 1)) < 3:
        N = 3 # number of the interfaces cannot be less than 3
    else:
        N = int(np.log(total_cross_prob) / np.log(p)) + 1
    # find the interfaces
    f_interval = ( x_flambda[-1] - x_flambda[0] ) / float(N-1)
    new_interfaces = []
    for i in range(N):
        value = f(x_flambda[0] + f_interval * i)
        new_interfaces.append(float(value))
    return new_interfaces
