"""Tests for the utility functions in the q2-gglasso plugin.

This module contains tests for various utility functions in the q2-gglasso plugin.
Currently, most tests are commented out and may be implemented in the future.
"""

import unittes
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
# test biom editing
