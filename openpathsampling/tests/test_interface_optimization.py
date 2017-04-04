from InterfaceOpt_class import new_interfaces_1, new_interfaces_2, get_interpolation_points
from nose.tools import raises, assert_equal
from numpy.testing import assert_allclose

class TestInterfaceOptimization(object):

    def test_function_1(self):
        interfaces = [0, 1, 2, 3]
        cross_prob = 0.5
        probabilities = [1., cross_prob, cross_prob, cross_prob]
        new_I = new_interfaces_1(interfaces, probabilities)
        assert_allclose(interfaces, new_I)

    def test_func1_exception(self):
        interfaces = [0, 2]
        cross_prob = [1.0, 0.3]
        new_int = new_interfaces_1(interfaces, cross_prob)
        assert_equal(3, len(new_int))
    
     
    def test_function_2(self):
        interfaces = [0, 2, 4, 6, 8, 10, 12]
        test_cross = 0.2
        prob = [1.]
        for item in interfaces[1:]:
            prob.append(test_cross)
        new_int_app_2 = new_interfaces_2(interfaces, prob, test_cross)
        N = len(new_int_app_2)
        interval = (interfaces[-1] - interfaces[0]) / (N-1)
        artificial_input = []
        for item in range(N):
            artificial_input.append(interfaces[0] + item * interval)
        assert_allclose(artificial_input, new_int_app_2)

    def test_function_2_int(self):
        interfaces = [0, 3, 6, 9, 12]
        test_cross = 0.8
        prob = [1.]
        for item in interfaces[1:]:
            prob.append(test_cross)
        new_int = new_interfaces_2(interfaces, prob, 0.6)
        middle = ( interface[-1] - interfaces[0] ) / 2.0
        artificial_input = [interfaces[0], middle, interfaces[-1]]     
        assert_equal(new_int, artificial_input)
