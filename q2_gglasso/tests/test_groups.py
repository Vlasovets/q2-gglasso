import unittest
import numpy as np
import pandas as pd
from biom.table import Table
from gglasso.helper.ext_admm_helper import create_group_array, construct_indexer, check_G

try:
    from q2_gglasso._func import build_groups

except ImportError:
    raise ImportWarning('Qiime2 not installed.')


class TestUtil(unittest.TestCase):

    def test_build_blocks(self):
        data1 = np.array([[0, 2, 1], [1, 1, 2], [2, 0, 2], [7, 0, 2]])
        sample_ids1 = ['S%d' % i for i in range(3)]
        observ_ids1 = ['OTU%d' % i for i in range(4)]

        data2 = np.array([[0, 2, 1], [1, 1, 2], [2, 0, 7], [7, 0, 3], [2, 0, 7], [7, 0, 3]])
        sample_ids2 = ['S%d' % i for i in range(3)]
        observ_ids2 = ['OTU%d' % i for i in range(6)]

        table_1 = Table(data1, observ_ids1, sample_ids1, table_id='Example Table1').transpose()
        table_2 = Table(data2, observ_ids2, sample_ids2, table_id='Example Table2').transpose()

        df1 = table_1.to_dataframe()
        df2 = table_2.to_dataframe()

        # GGLasso requires (p,n) input where p-variables, n-samples.
        df_list = [df1.T, df2.T]

        ix_exist, ix_location = construct_indexer(df_list)
        G = create_group_array(ix_exist, ix_location)

        t_list = [table_1, table_2]

        G_q2 = build_groups(t_list)

        if G.all() == G_q2.all():
            equal = True

        self.assertTrue(equal, msg="Group array of QIIME2 action is different from Group array of GGLasso")


if __name__ == '__main__':
    unittest.main()