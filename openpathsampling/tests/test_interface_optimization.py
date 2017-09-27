from openpathsampling.analysis.interface_optimization import (save_n_interfaces, 
                                                              save_p_interfaces, 
                                                              get_interpolation_points)
from nose.tools import raises, assert_equal
from numpy.testing import assert_allclose
import numpy.random as random

# nosetests --with-coverage --cover-erase --cover-package=interface_optimization --cover-html test_interface_optimization.py
#

class TestInterfaceOptimization(object):

    def test_function_n(self):
        interfaces = [0, 1, 2, 3]
        cross_prob = 0.5
        probabilities = [1., cross_prob, cross_prob, cross_prob]
        new_I = save_n_interfaces(interfaces, probabilities)
        assert_allclose(interfaces, new_I)


    def test_func_n_exception(self):
        interfaces = [0, 2]
        cross_prob = [1.0, 0.3]
        new_int = save_n_interfaces(interfaces, cross_prob)
        assert_equal(3, len(new_int))


    def test_func_n_null(self):
        interfaces = [0, 2, 4, 6]
        cross_prob = [1.0, 0.5, 0.5, 0.5]
        new_int = save_n_interfaces(interfaces, cross_prob, fixed_n=5)
        assert_equal(5, len(new_int))
    
     
    def test_function_p(self):
        interfaces = [0, 2, 4, 6, 8, 10, 12]
        test_cross = 0.2
        prob = [0.2] * len(interfaces)
        for i, item in enumerate(interfaces[1:]):
            prob[i + 1] = test_cross
        new_int_app_2 = save_p_interfaces(interfaces, prob, test_cross)
        N = len(new_int_app_2)
        interval = (interfaces[-1] - interfaces[0]) / float(N-1)
        artificial_input = [interfaces[0] + i * interval for i in range(N)]
        assert_allclose(artificial_input, new_int_app_2)


    def test_function_p_int(self):
        interfaces = [0, 3, 6, 9, 12]
        test_cross = 0.8
        prob = [1.] * len(interfaces)
        for i, item in enumerate(interfaces[1:]):
            prob[i + 1] = test_cross
        new_int = save_p_interfaces(interfaces, prob, 0.6)
        middle = (interfaces[-1] - interfaces[0]) / 2.0
        artificial_input = [interfaces[0], middle, interfaces[-1]]     
        assert_equal(new_int, artificial_input)


    def test_func_p_3(self):
        interfaces = [0, 2]
        cross_prob = [1.0, 0.9]
        new_int = save_p_interfaces(interfaces, cross_prob, fixed_p=0.1)
        print (new_int)
        assert_equal(3, len(new_int))


    def test_interpolation_points(self):
        interfaces = [0, 1, 2, 3, 4]
        crossing_probabilities = [0] * len(interfaces)
        for i, item in enumerate(interfaces):
            crossing_probabilities[i] = (random.random()+0.001)*0.99
        x, y = get_interpolation_points(interfaces, crossing_probabilities)
        # x has to be monotonicaly increasing array of values
        is_monotonic = True
        for i in range(len(x)-1):
            is_monotonic = (is_monotonic and (x[i] <= x[i + 1]))
        assert_equal(True, is_monotonic)
