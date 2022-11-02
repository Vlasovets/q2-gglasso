import unittest
import numpy as np
from biom.table import Table

try:
    from q2_gglasso.utils import zero_imputation

except ImportError:
    raise ImportWarning('Qiime2 not installed.')


class TestUtil(unittest.TestCase):
    def test_zero_imputation(self, pseudo_count=1, equal=False):
        data = np.array([[0, 0.1], [0.5, 0.1], [0.3, 0], [0.2, 0.8]])
        sample_ids = ['S%d' % i for i in range(2)]
        observ_ids = ['O%d' % i for i in range(4)]
        sample_metadata = [{'environment': 'A'}, {'environment': 'B'}]
        observ_metadata = [{'taxonomy': ['Bacteria', 'Firmicutes']},
                           {'taxonomy': ['Bacteria', 'Firmicutes']},
                           {'taxonomy': ['Bacteria', 'Proteobacteria']},
                           {'taxonomy': ['Bacteria', 'Proteobacteria']}]
        table = Table(data, observ_ids, sample_ids, observ_metadata, sample_metadata, table_id='Example Table')

        X = table.to_dataframe()
        X = X.sparse.to_dense()

        original_sum = X.sum(axis=0)
        X = zero_imputation(X)
        imputeted_sum = X.sum(axis=0)

        if (original_sum == imputeted_sum).all():
            equal = True

        self.assertTrue(equal, msg="Subcomposition coherence not fulfilled.")


if __name__ == '__main__':
    unittest.main()