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
from bokeh.layouts import gridplot, column, row, layout
from biom.table import Table
from bokeh.plotting import figure, curdoc
from biom.table import Table
from q2_gglasso.utils import PCA
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, Select, CustomJS
from bokeh.models import Panel, Tabs
from bokeh.palettes import Spectral6, Blues8

# solution = zarr.load("data/atacama_low/problem.zip")
# # # mapping = pd.read_csv("data/atacama-sample-metadata.tsv", sep='\t', index_col=0)
# df = pd.read_csv(str("data/atacama-table_clr_small/small_clr_feature-table.tsv"), index_col=0, sep='\t')
#

# depth = counts.sum(axis=1)
# df['depth'] = depth.values
# df = df.fillna(0)
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

# def create_figure(df, x, y, size, color):
#     SIZES = list(range(6, 22, 3))
#     COLORS = Spectral5
#     N_SIZES = len(SIZES)
#     N_COLORS = len(COLORS)
#
#     xs = df[x.value].values
#     ys = df[y.value].values
#     x_title = x.value.title()
#     y_title = y.value.title()
#
#     kw = dict()
#     kw['x_range'] = sorted(set(xs))
#     kw['y_range'] = sorted(set(ys))
#     kw['title'] = "%s vs %s" % (x_title, y_title)
#
#     p = figure(height=600, width=800, tools='pan,box_zoom,hover,reset',
#                x_axis_label=x.value, y_axis_label=y.value,
#                tooltips=[(x.value, "@" + y.value),
#                          (x.value, "@" + y.value)
#                          ],
#                title=x_title + " vs " + y_title
#                )
#     p.xaxis.axis_label = x_title
#     p.yaxis.axis_label = y_title
#
#     p.xaxis.major_label_orientation = np.pi / 4
#
#     sz = 9
#     if size.value != 'None':
#         if len(set(df[size.value])) > N_SIZES:
#             groups = pd.qcut(df[size.value].values, N_SIZES, duplicates='drop')
#         else:
#             groups = pd.Categorical(df[size.value])
#         sz = [SIZES[xx] for xx in groups.codes]
#
#     c = "#31AADE"
#     if color.value != 'None':
#         if len(set(df[color.value])) > N_COLORS:
#             groups = pd.qcut(df[color.value].values, N_COLORS, duplicates='drop')
#         else:
#             groups = pd.Categorical(df[color.value])
#         c = [COLORS[xx] for xx in groups.codes]
#
#     source = ColumnDataSource(data=dict(xs=xs, ys=ys))
#
#     p.circle(x='xs', y='ys', color=c, size=sz, source=source,
#              line_color="white", alpha=0.6, hover_color='white', hover_alpha=0.5)
#
#     return p
#
#
# def update(attr, old, new):
#     layout.children[1] = create_figure()
#
# def project_covariates(counts=pd.DataFrame(), metadata=pd.DataFrame(), L=np.ndarray):
#     proj, loadings, eigv = PCA(counts.dropna(), L, inverse=True)
#     r = np.linalg.matrix_rank(L)
#     pc_columns = list('PC_{0}'.format(i) for i in range(1, r + 1))
#     df_proj = pd.DataFrame(proj, columns=pc_columns, index=counts.index)
#
#     df = df_proj.join(metadata)
#     depth = counts.sum(axis=1)
#     df['depth'] = depth.values
#     df = df.fillna(0)
#
#     columns = df.columns
#
#     select_x = Select(title='X-Axis', value='PC_1', options=list(columns))
#     select_y = Select(title='Y-Axis', value='PC_2', options=list(columns))
#     select_size = Select(title='Size', value='None', options=['None'] + list(columns))
#     select_color = Select(title='Color', value='None', options=['None'] + list(columns))
#
#     p = create_figure(df=df, x=select_x, y=select_y, size=select_size, color=select_color)
#
#     # ax1 = p.xaxis, ax2 = p.yaxis
#
#     changeVariables = CustomJS(
#         args=dict(plot=p, select1=select_x, select2=select_y), code="""
#         var x = select_x.value;
#         console.log(x)
#         var y = select_y.value;
#         console.log(y)
#         plot.title.text = x + " vs " + y;
#         ax1[0].axis_label = x;
#         ax2[0].axis_label = y;
#         source.data['x'] = source.data[x];
#         source.data['y'] = source.data[y];
#         source.change.emit();
#     """)
#
#     select_x.js_on_change('value', changeVariables)
#     select_y.js_on_change('value', changeVariables)
#     # select_size.js_on_change('value', changeVariables)
#     # select_color.js_on_change('value', changeVariables)
#
#     controls = column(select_x, select_y, width=200)
#     layout = row(controls, p)
#
#     return layout


def project_covariates(counts=pd.DataFrame(), metadata=pd.DataFrame(), L=np.ndarray, color_by=str):
    proj, loadings, eigv = PCA(counts.dropna(), L, inverse=True)
    r = np.linalg.matrix_rank(L)
    pc_columns = list('PC{0}'.format(i) for i in range(1, r + 1))
    df_proj = pd.DataFrame(proj, columns=pc_columns, index=counts.index)

    df = df_proj.join(metadata)

    varName1 = 'PC1'
    varName2 = 'PC2'
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

    exp_cmap = LinearColorMapper(palette=Blues8, low=min(df[color_by].values), high=max(df[color_by].values))
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
    color_bar_plot = figure(title=title, title_location="right",
                            height=50, width=50, toolbar_location=None, min_border=0,
                            outline_line_color=None)

    bar = ColorBar(color_mapper=color_map, location=(1, 1))

    color_bar_plot.add_layout(bar, 'right')
    color_bar_plot.title.align = "center"
    color_bar_plot.title.text_font_size = '12pt'

    return color_bar_plot


def make_plots(df: pd.DataFrame, col_name: str = None, n_components=int, proj=np.ndarray, eigv=np.ndarray):
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
    # TO DO: Add widget for columns selection
    plot_dict = dict()
    for col in metadata.columns:
        plot_df = counts.join(metadata[col])
        plot_df = plot_df.dropna()

        proj, loadings, eigv = PCA(plot_df.iloc[:, :-1], L, inverse=True)
        r = np.linalg.matrix_rank(L)

        assert n_components < r, f"n_components is greater than the rank, got: {n_components}"

        pca_plot = make_plots(df=plot_df, col_name=col, n_components=n_components, proj=proj, eigv=eigv)

        plot_dict[col] = pca_plot

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
    J_ENV = jinja2.Environment(
        loader=jinja2.PackageLoader('q2_gglasso._pca', 'assets')
    )
    template = J_ENV.get_template('index.html')
    df = table.to_dataframe()
    L = solution['solution/lowrank_']

    numeric_md_cols = sample_metadata.filter_columns(column_type='numeric')
    md = numeric_md_cols.to_dataframe()
    md = md.reindex(df.index)

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
