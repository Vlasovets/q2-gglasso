import unittest
import warnings
from gglasso.helper.data_generation import generate_precision_matrix, sample_covariance_matrix

from gglasso.problem import glasso_problem

try:
    from q2_gglasso._func import solve_problem

except ImportError:
    raise ImportWarning('Qiime2 not installed.')

# class TestUtil(unittest.TestCase):
#
#
#
# if __name__ == '__main__':
#     unittest.main()
#

# def test_assertWarns(self):
#     with self.assertWarnsRegex(Warning, 'The solution might have not reached global minimum!'):
#         warnings.warn("The solution might have not reached global")
### test biom editing

