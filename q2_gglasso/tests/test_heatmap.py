import unittest
import numpy as np
import pandas as pd
import zarr

try:
    from q2_gglasso._summarize._visualizer import _make_heatmap, _get_order, _get_labels, \
        hierarchical_clustering
except ImportError:
    raise ImportWarning('Qiime2 not installed.')


class TestUtil(unittest.TestCase):
    def test_heatmap(self):

        sol = zarr.load(store="../../data/atacama-solution-adapt/problem.zip")
        sol = zarr.load(store="data/atacama-solution-adapt/problem.zip")

        labels_dict, labels_dict_reversed = _get_labels(solution=sol, clustered=False)

        S = pd.DataFrame(sol['covariance'])

        df = pd.DataFrame(S.stack(), columns=['covariance']).reset_index()
        df.columns = ["taxa_y", "taxa_x", "covariance"]
        df = df.replace({"taxa_x": labels_dict, "taxa_y": labels_dict})

        clust_order = _get_order(S, method='average', metric='euclidean')

        asv_part = S.iloc[:-4, :-4]

        clust_order = _get_order(asv_part, method='average', metric='euclidean')

        labels_dict_new = {i: labels_dict[key] for i, key in enumerate(clust_order)}
        labels_covs = {k: v for k, v in labels_dict.items() if k >= len(clust_order)}
        labels_dict_new.update(labels_covs)

        labels_dict_reversed = {len(labels_dict_new) - 1 - k: v for k, v in labels_dict_new.items()}



        len(clust_order)

        labels_dict_1 = {i: labels_dict[key] for i, key in enumerate(clust_order)}

        re_asv_part = asv_part.iloc[clust_order, clust_order]

        cov_asv_part = S.iloc[:-4, -4:].iloc[clust_order, :]
        cov_part = S.iloc[-4:, -4:]

        res = np.block([[re_asv_part.values, cov_asv_part.values],
                        [cov_asv_part.T.values, cov_part.values]])

        labels = list(re_asv_part.columns) + list(cov_part.columns)
        re_data = pd.DataFrame(res, index=labels, columns=labels)

        res = np.block([[re_asv_part.values, cov_asv_part.values],
                        [cov_asv_part.T.values, cov_part.values]])



        S_cl = hierarchical_clustering(S, clust_order=clust_order, n_covariates=2)

        # S_cl.columns = range(S_cl.columns.size)
        # S_cl = S_cl.reset_index(drop=True)


        # reset_dict = {i: labels_dict[key] for i, key in enumerate(clust_order)}
        # reindexed_dict = {len(reset_dict) - 1 - k: v for k, v in reset_dict.items()}
        #
        # labels_dict = {i: labels_dict[key] for i, key in enumerate(clust_order)}
        # labels_dict_reversed = {len(labels_dict) - 1 - k: v for k, v in labels_dict.items()}
        #
        # shifted_labels_dict = {k + 0.5: v for k, v in labels_dict.items()}
        #
        # S_cl = S_cl.T.reset_index(drop=True).T.reset_index(drop=True)
        #
        # df_cl = pd.DataFrame(S_cl.stack(), columns=['covariance']).reset_index()
        # df_cl.columns = ["taxa_y", "taxa_x", "covariance"]
        # df_cl = df_cl.replace({"taxa_x": labels_dict, "taxa_y": labels_dict})

        # p1 = _make_heatmap(data=S, title="Sample covariance",
        #                    width=500, height=50,
        #                    label_size="25pt", labels_dict=labels_dict,
        #                    labels_dict_reversed=labels_dict_reversed)

if __name__ == '__main__':
    unittest.main()
