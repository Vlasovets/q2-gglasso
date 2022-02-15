import pandas as pd
import qiime2

from q2_gglasso.plugin_setup import plugin
from q2_gglasso._format import GGLassoDataFormat

@plugin.register_transformer
def _1(data: pd.DataFrame) -> GGLassoDataFormat:
    ff = GGLassoDataFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', index_label=('feature1', 'feature2'))
    return ff

@plugin.register_transformer
def _2(ff: GGLassoDataFormat) -> pd.DataFrame:
    df = pd.read_table(str(ff), index_col=(0, 1), sep='\t')
    new_index = pd.MultiIndex.from_tuples([(str(i), str(j)) for i, j in df.index])
    df.index = new_index
    return df