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

    def test_build_blocks(self):
        data1 = np.array([[0, 2, 1], [1, 1, 2], [2, 0, 2], [7, 0, 2]])
        sample_ids1 = ['S%d' % i for i in range(3)]
        observ_ids1 = ['OTU%d' % i for i in range(4)]

        data2 = np.array([[0, 2, 1], [1, 1, 2], [2, 0, 7], [7, 0, 3], [2, 0, 7], [7, 0, 3]])
        sample_ids2 = ['S%d' % i for i in range(3)]
        observ_ids2 = ['OTU%d' % i for i in range(6)]

        table_1 = Table(data1, observ_ids1, sample_ids1, table_id='Example Table1').transpose()
        table_2 = Table(data2, observ_ids2, sample_ids2, table_id='Example Table2').transpose()

        table_1 = load_table('data/atacama-table_mclr/feature-table.biom')
        table_2 = load_table('data/atacama-supertable_clr/feature-table.biom')

        df1 = table_1.to_dataframe()
        df2 = table_2.to_dataframe()

        # GGLasso requires (p,n) input where p-variables, n-samples.
        df_list = [df1.T, df2.T]

        ix_exist, ix_location = construct_indexer(df_list)
        G = create_group_array(ix_exist, ix_location)

        check_G(G, [df1.shape[1], df2.shape[1]])

        tables = [table_1, table_2]

        columns_dict = dict()
        dataframes_p_N = list()
        p_arr = list()
        num_samples = list()

        i = 0
        for table in tables:
            df = table.to_dataframe()

            dataframes_p_N.append(df.T)  # (p_variables, N_samples) required shape of dataframe
            p_arr.append(df.shape[1])  # number of variables
            num_samples.append(df.shape[0])  # number of samples

            columns_dict[i] = df.columns.values.tolist()
            i += 1

        all_names = set()
        for columns in columns_dict.values():
            for name in columns:
                all_names.add(name)

        non_conforming_problem = False

        for k in range(0, len(columns_dict)):
            diff = all_names.difference(columns_dict[k])
            if len(diff) > 0:
                non_conforming_problem = True

        if non_conforming_problem:
            # tables_trans = transpose_dataframes(dataframes)

            ix_exist, ix_location = construct_indexer(dataframes_p_N)

            G = create_group_array(ix_exist, ix_location)

        G_q2 = build_groups(tables)

        if G.all() == G_q2.all():
            equal = True

        self.assertTrue(equal, msg="Group array of QIIME2 action is different from Group array of GGLasso")


if __name__ == '__main__':
    unittest.main()