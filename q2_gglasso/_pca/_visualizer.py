import zarr
import itertools
import numpy as np
import pandas as pd
import qiime2
import os
import jinja2
import warnings

from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.layouts import gridplot, column, row, layou
from biom.table import Table
from bokeh.plotting import figure, curdoc
from q2_gglasso.utils import PCA, get_seq_depth
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, Select, CustomJS
from bokeh.models import Panel, Tabs
from bokeh.palettes import Spectral6, Blues8

# solution = zarr.load("data/atacama_low/problem.zip")
# # # mapping = pd.read_csv("data/atacama-sample-metadata.tsv", sep='\t', index_col=0)
# df = pd.read_csv(str("data/atacama-table_clr_small/small_clr_feature-table.tsv"), index_col=0, sep='\t')
#
#
# # depth = counts.sum(axis=1)
# # df['depth'] = depth.values
# # df = df.fillna(0)
# #
# # df.columns
# #
# # from qiime2 import Artifact, sdk
# #
# # # # # # taxonomy = Artifact.load('data/classification.qza')
# # table = Artifact.load('data/atacama-table_clr.qza')
# # table = table.view(Table)
# #
# sample_metadata = qiime2.Metadata.load("data/atacama-sample-metadata.tsv")
# counts = df


def project_covariates(counts=pd.DataFrame(), metadata=pd.DataFrame(), L=np.ndarray, color_by=str):
    """Project data onto principal component space with covariates.

    Parameters
    ----------
    counts : pd.DataFrame
        The count data to project.
    metadata : pd.DataFrame
        The metadata containing covariates.
    L : np.ndarray
        Low rank component from the GGLasso solution.
    color_by : str
        The metadata column to use for coloring the projection.

    Returns
    -------
    bokeh.layouts.row
        A Bokeh layout containing the PCA projection plot.
    """
    proj, loadings, eigv = PCA(counts.dropna(), L, inverse=True)
    r = np.linalg.matrix_rank(L)
    eigv_sum = np.sum(eigv)
    var_exp = [(value / eigv_sum) for value in sorted(eigv, reverse=True)]

    pc_columns = list('PC{0} ({1}%)'.format(i + 1, str(100 * var_exp[i])[:4]) for i in range(0, r))
    df_proj = pd.DataFrame(proj, columns=pc_columns, index=counts.index)
    df = df_proj.join(metadata)

    varName1 = 'PC1 ({0}%)'.format(str(100 * var_exp[0])[:4])
    varName2 = 'PC2 ({0}%)'.format(str(100 * var_exp[1])[:4])
    df['x'] = df[varName1]
    df['y'] = df[varName2]

    source = ColumnDataSource(df)

    p0 = figure(tools='save, zoom_in, zoom_out, wheel_zoom, box_zoom, reset', plot_width=800, plot_height=800,
                active_scroll="wheel_zoom",
                x_axis_label=varName1, y_axis_label=varName2,
                tooltips=[(varName1, "@" + varName1),
                          (varName2, "@" + varName2)
                          ],
                title=varName1 + " vs " + varName2)

    exp_cmap = LinearColorMapper(palette=Blues8[::-1], low=min(df[color_by].values), high=max(df[color_by].values))
    p0.circle('x', 'y', source=source, size=15, line_color=None, fill_color={"field": color_by, "transform": exp_cmap},
              fill_alpha=0.3)

    options = list(df.columns)
    options.remove('x')
    options.remove('y')

    select1 = Select(title="X-Axis:", value=varName1, width=100, options=options)
    select2 = Select(title="Y-Axis:", value=varName2, width=100, options=options)

    changeVariables = CustomJS(
        args=dict(plot=p0, source=source, select1=select1, select2=select2, ax1=p0.xaxis, ax2=p0.yaxis), code="""
        var varName1 = select1.value;
        var varName2 = select2.value;
        plot.title.text = varName1 + " vs " + varName2;
        ax1[0].axis_label = varName1;
        ax2[0].axis_label = varName2;
        source.data['x'] = source.data[varName1];
        source.data['y'] = source.data[varName2];
        source.change.emit();
    """)

    select1.js_on_change("value", changeVariables)
    select2.js_on_change("value", changeVariables)

    color_bar_plot = figure(title=color_by, title_location="right",
                            height=500, width=150, toolbar_location=None, min_border=0,
                            outline_line_color=None)

    bar = ColorBar(color_mapper=exp_cmap, location=(1, 1))

    color_bar_plot.add_layout(bar, 'right')
    color_bar_plot.title.align = "center"
    color_bar_plot.title.text_font_size = '12pt'

    layout_1 = row(p0, color_bar_plot, select1, select2)

    return layout_1


def add_color_bar(color_map: LinearColorMapper, title: str = None):
    """Create a color bar for the plot.

    Parameters
    ----------
    color_map : LinearColorMapper
        The color mapper to use for the color bar.
    title : str, optional
        The title of the color bar.

    Returns
    -------
    bokeh.plotting.figure
        A figure containing the color bar.
    """
    color_bar_plot = figure(title=title, title_location="right",
                            height=50, width=50, toolbar_location=None, min_border=0,
                            outline_line_color=None)

    bar = ColorBar(color_mapper=color_map, location=(1, 1))

    color_bar_plot.add_layout(bar, 'right')
    color_bar_plot.title.align = "center"
    color_bar_plot.title.text_font_size = '12pt'

    return color_bar_plo


def make_plots(df: pd.DataFrame, col_name: str = None, n_components=int, proj=np.ndarray, eigv=np.ndarray):
    """Create a grid of PCA plots.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    col_name : str, optional
        The column name to use for coloring.
    n_components : in
        The number of principal components to plot.
    proj : np.ndarray
        The projected data points.
    eigv : np.ndarray
        The eigenvalues for the principal components.

    Returns
    -------
    bokeh.layouts.gridplo
        A grid of PCA projection plots.
    """
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
    layout_2 = gridplot(children=list(x), ncols=int(n_components), width=250, height=250)

    return layout_2


def pair_plot(counts: pd.DataFrame(), metadata=pd.DataFrame(), L=np.ndarray, n_components=int, color_by=str):
    """Create a pair plot of PCA projections.

    Parameters
    ----------
    counts : pd.DataFrame
        The count data to project.
    metadata : pd.DataFrame
        The metadata containing covariates.
    L : np.ndarray
        Low rank component from the GGLasso solution.
    n_components : in
        The number of principal components to include.
    color_by : str
        The metadata column to use for coloring the projection.

    Returns
    -------
    bokeh.layouts.row
        A row layout containing the pair plot.
    """
    # TO DO: Add widget for columns selection
    plot_dict = dict()
    for col in metadata.columns:
        plot_df = counts.join(metadata[col])
        plot_df = plot_df.dropna()

        proj, loadings, eigv = PCA(plot_df.iloc[:, :-1], L, inverse=True)
        r = np.linalg.matrix_rank(L)

        assert n_components < r, f"n_components is greater than the rank, got: {n_components}"

        pca_plot = make_plots(df=plot_df, col_name=col, n_components=n_components, proj=proj, eigv=eigv)

        plot_dict[col] = pca_plo

    if color_by is None:
        warnings.warn("Coloring covariate has not been selected, "
                      "the first entry from the following list will be used:{0}".format(metadata.columns))
        color_by = metadata.columns[0]
        p2 = row(plot_dict[color_by])
    else:
        p2 = row(plot_dict[color_by])

    return p2


def pca(output_dir: str, table: Table, solution: zarr.hierarchy.Group, n_components: int = 3, color_by: str = None,
        sample_metadata: qiime2.Metadata = None):
    """Generate PCA visualization from GGLasso solution.

    Parameters
    ----------
    output_dir : str
        The directory where visualization files will be written.
    table : Table
        The feature table containing sample data.
    solution : zarr.hierarchy.Group
        The GGLasso solution containing the low-rank component.
    n_components : int, optional
        The number of principal components to display, by default 3.
    color_by : str, optional
        The metadata column to use for coloring points, by default None.
    sample_metadata : qiime2.Metadata, optional
        The sample metadata, by default None.
    """
    J_ENV = jinja2.Environment(
        loader=jinja2.PackageLoader('q2_gglasso._pca', 'assets')
    )
    template = J_ENV.get_template('index.html')
    df = table.to_dataframe()
    L = solution['solution/lowrank_']

    numeric_md_cols = sample_metadata.filter_columns(column_type='numeric')
    md = numeric_md_cols.to_dataframe()
    md = md.reindex(df.index)

    if color_by is None:
        depth = get_seq_depth(df)
        depth = pd.DataFrame(depth, index=depth.index, columns=["seq-depth"])
        md = md.join(depth)
        color_by = "seq-depth"

    p1 = project_covariates(counts=df, metadata=md, L=L, color_by=color_by)
    p2 = pair_plot(counts=df, metadata=md, L=L, n_components=n_components, color_by=color_by)

    tab1 = Panel(child=p1, title="Single plot")
    tab2 = Panel(child=p2, title="Pair-plot")

    tabs = [tab1, tab2]
    p = Tabs(tabs=tabs)

    script, div = components(p, INLINE)
    output_from_parsed_template = template.render(plot_script=script, plot_div=div)

    with open(os.path.join(output_dir, 'index.html'), 'w') as fh:
        fh.write(output_from_parsed_template)
