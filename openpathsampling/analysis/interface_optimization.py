"""This module defined to collect function needed for interface positions optimization.

Important methods defined here
------------------------------

find_interface_and_cross
    Reads the output of the analyze-tool after a certain number of cycles
    (defined by the user). Information about 'transition' must be provided.

get_interpolation_points:
    A method to calculate the points of the transformation, that is going to map required
    sequence to the new values of the interface positions.

save_n_interfaces
    A method for calculation of the new interface positions, using I approach:
    the number of interfaces kept the same, only the values of the positions are
    going to be changed.

save_p_interfaces
    A method for calculation of the new interface positions, using II approach: the number of
    the interfaces might be changed, "expected" crossing probability for each ensemble has a
    predefined value.

"""
import numpy as np
from scipy.interpolate import interp1d

def find_interface_and_cross(transition):  # pragma: no cover
    """To read the output of the analysis tool, and get interface positions
    and crossing probabilities. Example to get input:
    transition = system.transitions[(stateA, stateB)]

    Parameters
    ----------
    transition : an object like :py:class:`TISTransition`
        Information about transition

    Returns
    -------
    old_interfaces : list of floats
        Positions of the interfaces in the simulation
    old_crossing_probabilities : list of floats
        Crossing probabilities for each ensemble
    """
    try:
        trans_hists = transition.histograms['max_lambda']
        all_hist_functions = {}
        old_interfaces = []
        old_probabilities = []
        for ensemble in trans_hists:
            hist_function = trans_hists[ensemble].reverse_cumulative()
            all_hist_functions[ensemble.interface.lambda_max] = hist_function
        sorted_keys = sorted(all_hist_functions.keys())
        for i, lambdai in enumerate(sorted_keys):
            prob_func = all_hist_functions[sorted_keys[i - 1]]
            prob = prob_func(lambdai)
            old_interfaces.append(lambdai)
            old_probabilities.append(prob)
        return old_interfaces, old_probabilities
    except TypeError:
        msg = 'Inappropriate input: transition should be an obj like `TISTransition`'
        print (msg)


def get_interpolation_points(pos_interfaces, cross_probabilities):
    """Calculated the point for transformation function.

    Parameters
    ----------
    pos_interfaces : list of floats
        Old interface positions
    cross_probabilities : list of floats
        Old crossing probabilities for each ensemble

    Returns
    -------
    out[0] : list of floats
        x-values for interpolation, related to the crossing probabilities
    out[1] : list of floats
        y-values for interpolation, related to the interfaces.
    """
    logarithms = np.log(cross_probabilities)
    point_values = []
    for i in range(len(pos_interfaces)):
        point_values.append(sum(logarithms[:(i + 1)]) / float(sum(logarithms)))
    return [point_values, pos_interfaces]


def _round(value, formatting):
    """Changes the format the value accordingly to the string formatting type.
    Basically used to reduce precision in defining of the number.

    Parameters
    ----------
    value : float or int
        Floating point number to change
    formatting : string
        String-type python keyword to define a formatting of the number,
        like '.4g', '.2f' etc.

    Returns
    -------
    out : float
        The value in the desired format.
    """
    result = float(format(value, formatting))
    return result


def _mapping_f(interfaces, cross_probabilities):
    """Defines a mapping function for the transformation.
    It should be monotonically increasing.

    Parameters
    ----------
    interfaces : list of floats
        Current interfaces from the simulation.
    cross_probabilities : list of floats
        Current crossing probabilities in the simulation. All values should be non-zero.

    Returns
    -------
    out : func
        A mapping function"""
    prob_points, interface_points = get_interpolation_points(interfaces, cross_probabilities)
    return interp1d(prob_points, interface_points, kind='linear')


def save_n_interfaces(interfaces, cross_probabilities, fixed_n=None, formatting='.4g'):
    """Calculates the new interfaces, if the number of the interfaces remains the same.

    Parameters
    ----------
    interfaces : list of floats
        Old interface positions
    cross_probabilities : list of floats
        Old crossing probabilities for each ensemble

    Returns
    -------
    out : list of floats
        New positions of the interfaces
    """
    prob_points = get_interpolation_points(interfaces, cross_probabilities)[0]
    # build transformation function
    mapping_f = _mapping_f(interfaces, cross_probabilities)
    result = []
    if fixed_n is None:
        number_of_interfaces = len(interfaces)
    else:
        number_of_interfaces = fixed_n
    if number_of_interfaces < 3:
        number_of_interfaces = 3  # number of the interfaces cannot be less than 3
    prob_step = (prob_points[-1] - prob_points[0]) / float(number_of_interfaces - 1)
    for i in range(number_of_interfaces):
        if i == 0:
            value = mapping_f(prob_points[0])
        elif i == (number_of_interfaces - 1):
            value = mapping_f(prob_points[-1])
        else:
            arg = prob_points[0] + prob_step * i
            value = mapping_f(arg)
        result.append(_round(float(value), formatting))
    return result


def save_p_interfaces(interfaces, cross_probabilities, fixed_p=0.4, formatting='.4g'):
    """Calculates the new interfaces, if the crossing probabilities
    expected to have some fixed value.

    Parameters
    ----------
    interfaces : list of floats
        Old interface positions
    cross_probabilities : list of floats
        Old crossing probabilities for each ensemble
    fixed_p : float
        Expected value for crossing probabilities in each ensemble. Default is 0.4.

    Returns
    -------
    out : list of floats
        New positions of the interfaces
    """
    prob_points = get_interpolation_points(interfaces, cross_probabilities)[0]
    # build interpolation function
    mapping_f = _mapping_f(interfaces, cross_probabilities)
    result = []
    total_cross_prob = 1
    for item in cross_probabilities:
        total_cross_prob = total_cross_prob * item
    number_of_interfaces = int(np.log(total_cross_prob) / np.log(fixed_p)) + 2
    if number_of_interfaces < 3:
        number_of_interfaces = 3  # number of the interfaces cannot be less than 3
    prob_step = (prob_points[-1] - prob_points[0]) / float(number_of_interfaces - 1)
    for i in range(number_of_interfaces):
        if i == 0:
            value = mapping_f(prob_points[0])
        elif i == (number_of_interfaces - 1):
            value = mapping_f(prob_points[-1])
        else:
            arg = prob_points[0] + prob_step * i
            value = mapping_f(arg)
        result.append(_round(float(value), formatting))
    return result
