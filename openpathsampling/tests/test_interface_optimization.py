from interface_optimization import new_interfaces_A, new_interfaces_B, get_interpolation_points
from nose.tools import raises, assert_equal
from numpy.testing import assert_allclose


# nosetests --with-coverage --cover-erase --cover-package=interface_optimization --cover-html test_interface_optimization.py
#

class TestInterfaceOptimization(object):

    def test_function_A(self):
        interfaces = [0, 1, 2, 3]
        cross_prob = 0.5
        probabilities = [1., cross_prob, cross_prob, cross_prob]
        new_I = new_interfaces_A(interfaces, probabilities)
        assert_allclose(interfaces, new_I)

    def test_funcA_exception(self):
        interfaces = [0, 2]
        cross_prob = [1.0, 0.3]
        new_int = new_interfaces_A(interfaces, cross_prob)
        assert_equal(3, len(new_int))
    
     
    def test_function_B(self):
        interfaces = [0, 2, 4, 6, 8, 10, 12]
        test_cross = 0.2
        prob = [1.]
        for item in interfaces[1:]:
            prob.append(test_cross)
        new_int_app_2 = new_interfaces_B(interfaces, prob, test_cross)
        N = len(new_int_app_2)
        interval = (interfaces[-1] - interfaces[0]) / (N-1)
        artificial_input = []
        for item in range(N):
            artificial_input.append(interfaces[0] + item * interval)
        assert_allclose(artificial_input, new_int_app_2)

    def test_function_B_int(self):
        interfaces = [0, 3, 6, 9, 12]
        test_cross = 0.8
        prob = [1.]
        for item in interfaces[1:]:
            prob.append(test_cross)
        new_int = new_interfaces_B(interfaces, prob, 0.6)
        middle = ( interfaces[-1] - interfaces[0] ) / 2.0
        artificial_input = [interfaces[0], middle, interfaces[-1]]     
        assert_equal(new_int, artificial_input)

    def test_interpolation_points(self):
        interfaces = [0, 1, 2, 3, 4]
        crossing_probabilities = [0.8, 0.7, 0.9, 0.5, 0.2]
        x, y = get_interpolation_points(interfaces, crossing_probabilities)

        interfaces = [0, 1, 2, 3, 4]
        crossing_probabilities = []
        import numpy.random as random
        for item in interfaces:
            crossing_probabilities.append((random.random()+0.001)*0.99)
        x, y = get_interpolation_points(interfaces, crossing_probabilities)
        # x has to be monotonicaly increasing array of values
        is_monotonic = True
        for i in range(len(x)-1):
            is_monotonic = ( is_monotonic and ( x[i] <= x[i + 1] ) )        
        assert_equal(True, is_monotonic)
