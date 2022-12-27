import zarr
import itertools
import numpy as np
import pandas as pd
import qiime2
import os
import jinja2
import warnings
import zarr

from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.layouts import gridplot, row
from biom.table import Table
from bokeh.plotting import figure
from biom.table import Table
from q2_gglasso.utils import PCA
from bokeh.models import ColumnDataSource
from bokeh.models import LinearColorMapper, ColorBar

# solution = zarr.load("data/atacama_low/problem.zip")
# # # mapping = pd.read_csv("data/atacama-sample-metadata.tsv", sep='\t', index_col=0)
# df = pd.read_csv(str("data/atacama-table_clr_small/small_clr_feature-table.tsv"), index_col=0, sep='\t')
#
# df.columns
#
# from qiime2 import Artifact, sdk
#
# # # # # taxonomy = Artifact.load('data/classification.qza')
# table = Artifact.load('data/atacama-table_clr.qza')
# table = table.view(Table)
#
# sample_metadata = qiime2.Metadata.load("data/atacama-sample-metadata.tsv")

def add_color_bar(color_map: LinearColorMapper, title: str = None):
    color_bar_plot = figure(title=title, title_location="right",
                            height=50, width=50, toolbar_location=None, min_border=0,
                            outline_line_color=None)

    bar = ColorBar(color_mapper=color_map, location=(1, 1))

    color_bar_plot.add_layout(bar, 'right')
    color_bar_plot.title.align = "center"
    color_bar_plot.title.text_font_size = '12pt'

    return color_bar_plot


def make_plots(df: pd.DataFrame, col_name: str = None, n_components: int = None, proj=np.ndarray, eigv=np.ndarray):
    t = np.arange(n_components)
    comb = list(itertools.combinations(t, 2))

    bokeh_tools = ["save, zoom_in, zoom_out, wheel_zoom, box_zoom, reset, hover"]

    grid = np.empty([n_components, n_components], dtype=object)
    eigv_sum = np.sum(eigv)
    var_exp = [(value / eigv_sum) for value in sorted(eigv, reverse=True)]

    for i, j in comb:
        p = figure(tools=bokeh_tools, toolbar_location='above')

        source = ColumnDataSource({'x': proj[:, i],
                                   'y': proj[:, j],
                                   'col': df[col_name].values})

        exp_cmap = LinearColorMapper(palette="Viridis256",
                                     low=min(df[col_name].values),
                                     high=max(df[col_name].values))

        p.circle("x", "y", size=5, source=source, line_color=None,
                 fill_color={"field": "col", "transform": exp_cmap})

        p.xaxis.axis_label = 'PC{0} ({1}%)'.format(i + 1, str(100 * var_exp[i])[:4])
        p.yaxis.axis_label = 'PC{0} ({1}%)'.format(j + 1, str(100 * var_exp[j])[:4])

        grid[i][j] = p

        grid[-1][-1] = add_color_bar(color_map=exp_cmap, title=col_name)

    x = np.transpose(grid)
    x = x.flatten()
    pair_plot = gridplot(children=list(x), ncols=int(n_components), width=250, height=250)

    return pair_plot


def pca(output_dir: str, table: Table, solution: zarr.hierarchy.Group, n_components: int = 3, color_by: str = None,
        sample_metadata: qiime2.Metadata = None):
    J_ENV = jinja2.Environment(
        loader=jinja2.PackageLoader('q2_gglasso._pca', 'assets')
    )

    template = J_ENV.get_template('index.html')

    df = table.to_dataframe()
    L = solution['solution/lowrank_']

    numeric_md_cols = sample_metadata.filter_columns(column_type='numeric')
    md = numeric_md_cols.to_dataframe()
    md = md.reindex(df.index)

    # TO DO: Add widget for columns selection
    plot_dict = dict()
    for col in md.columns:
        plot_df = df.join(md[col])
        plot_df = plot_df.dropna()

        proj, loadings, eigv = PCA(plot_df.iloc[:, :-1], L, inverse=True)
        r = np.linalg.matrix_rank(L)

        assert n_components < r, f"n_components is greater than the rank, got: {n_components}"

        pca_plot = make_plots(df=plot_df, col_name=col, n_components=n_components, proj=proj, eigv=eigv)

        plot_dict[col] = pca_plot

    if color_by is None:
        warnings.warn("Coloring covariate has not been selected, "
                      "the first entry from the following list will be used:{0}".format(md.columns))
        plot = plot_dict[md.columns[0]]
    else:
        plot = plot_dict[color_by]

    script, div = components(plot, INLINE)
    output_from_parsed_template = template.render(plot_script=script, plot_div=div)

    with open(os.path.join(output_dir, 'index.html'), 'w') as fh:
        fh.write(output_from_parsed_template)
