import unittest
import numpy as np
import pandas as pd
from biom.table import Table
from biom import load_table
from gglasso.helper.ext_admm_helper import create_group_array, construct_indexer, check_G

try:
    from q2_gglasso._func import build_groups

except ImportError:
    raise ImportWarning('Qiime2 not installed.')


class TestUtil(unittest.TestCase):

    def test_build_blocks(self, equal=False):

        data1 = np.random.rand(4, 2)
        sample_ids1 = ['S%d' % i for i in range(2)]
        observ_ids1 = ['OTU%d' % i for i in range(4)]

        data2 = np.random.rand(6, 3)
        sample_ids2 = ['S%d' % i for i in range(3)]
        observ_ids2 = ['OTU%d' % i for i in range(6)]

        table_1 = Table(data1, observ_ids1, sample_ids1, table_id='Example Table1')
        table_2 = Table(data2, observ_ids2, sample_ids2, table_id='Example Table2')

        # table_1 = load_table('data/atacama-table_mclr/feature-table.biom')
        # table_2 = load_table('data/atacama-supertable_clr/feature-table.biom')

        df1 = table_1.to_dataframe()
        df2 = table_2.to_dataframe()
        p_list = [df1.shape[0], df2.shape[0]]

        # GGLasso requires (p,n) input where p-variables, n-samples.
        df_list = [df1, df2]

        ix_exist, ix_location = construct_indexer(df_list)
        G = create_group_array(ix_exist, ix_location)

        check_G(G, p_list)

        t_list = [table_1, table_2]

        G_q2 = build_groups(t_list)

        if G.all() == G_q2.all():
            equal = True

        self.assertTrue(equal, msg="Group array of QIIME2 action is different from Group array of GGLasso")


if __name__ == '__main__':
    unittest.main()
