import os
import zarr
import jinja2
import bisect
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from math import pi
from biom.table import Table
from itertools import chain
from q2_gglasso.utils import flatten_array, pep_metric, reset_columns_and_index

from bokeh.plotting import figure
from bokeh.models import HoverTool, Panel, Tabs, ColorBar, LinearColorMapper
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.palettes import RdBu
from bokeh.layouts import row, column


def _get_bounds(nlabels: int):
    bottom = list(chain.from_iterable([[ii] * nlabels for ii in range(nlabels)]))
    top = list(chain.from_iterable([[ii + 1] * nlabels for ii in range(nlabels)]))
    left = list(chain.from_iterable([list(range(nlabels)) for ii in range(nlabels)]))
    right = list(chain.from_iterable([list(range(1, nlabels + 1)) for ii in range(nlabels)]))

    return bottom, top, left, right


def _get_colors(df: pd.DataFrame()):
    rdbu = plt.get_cmap('RdBu')
    cmap = ListedColormap(rdbu(np.arange(256)))

    # Create a list of hex color codes from the colormap
    colors = [cmap(i)[:3] for i in range(256)]
    colors = ['#' + ''.join([format(int(c * 255), '02x') for c in color]) for color in colors]
    colors = colors[::-1]  # red - positive, blue - negative

    ccorr = np.arange(-1, 1, 1 / (len(colors) / 2))
    color_list = []
    for value in df.covariance.values:
        ind = bisect.bisect_left(ccorr, value)  # smart array insertion
        if ind == 0:  # avoid ind == -1 on the next step
            ind = ind + 1
        color_list.append(colors[ind - 1])
    return color_list, colors


def _get_labels(solution: zarr.hierarchy.Group, clustered: bool = False, clust_order: list = None):
    labels_dict = dict()
    labels_dict_reversed = dict()
    p = np.array(solution['p']).item()
    for i in range(0, p):
        labels_dict[i] = np.array(solution['labels/{0}'.format(i)]).item()
        labels_dict_reversed[p - 1] = np.array(solution['labels/{0}'.format(i)]).item()
        p -= 1

    return labels_dict, labels_dict_reversed


def _get_order(data: pd.DataFrame, method: str = 'average', metric: str = 'euclidean'):
    """
    Performs hierarchical clustering on the input DataFrame and returns the cluster order.

    Args:
        data (pd.DataFrame): The input DataFrame.
        method (str, optional): The clustering method. Defaults to 'average'.
        metric (str, optional): The distance metric. Defaults to 'euclidean'.

    Returns:
        list: The cluster order as a list of indices.
    """
    grid = sns.clustermap(data, method=method, metric=metric, robust=True)
    plt.close()
    clust_order = grid.dendrogram_row.reordered_ind

    return clust_order


def hierarchical_clustering(data: pd.DataFrame, clust_order: list, n_covariates: int = None):
    if n_covariates is None:
        re_data = data.iloc[clust_order, clust_order]

    else:
        asv_part = data.iloc[:-n_covariates, :-n_covariates]
        re_asv_part = asv_part.iloc[clust_order, clust_order]
        cov_asv_part = data.iloc[:-n_covariates, -n_covariates:].iloc[clust_order, :]
        cov_part = data.iloc[-n_covariates:, -n_covariates:]

        res = np.block([[re_asv_part.values, cov_asv_part.values],
                        [cov_asv_part.T.values, cov_part.values]])

        labels = list(re_asv_part.columns) + list(cov_part.columns)
        re_data = pd.DataFrame(res, index=labels, columns=labels)

    return re_data


def _make_heatmap(data: pd.DataFrame(), title: str = None, labels_dict: dict = None,
                  labels_dict_reversed: dict = None,
                  width: int = 1500, height: int = 1500, label_size: str = "5pt",
                  title_size: str = "24pt", not_low_rank: bool = True):
    nlabels = len(labels_dict)
    shifted_labels_dict = {k + 0.5: v for k, v in labels_dict.items()}
    shifted_labels_dict_reversed = {k + 0.5: v for k, v in labels_dict_reversed.items()}

    df = data.iloc[::-1]  # rotate the diagonal
    df = pd.DataFrame(df.stack(), columns=['covariance']).reset_index()
    df.columns = ["taxa_y", "taxa_x", "covariance"]
    # clustering is correct, check the labels
    df = df.replace({"taxa_x": labels_dict, "taxa_y": labels_dict})

    color_list, colors = _get_colors(df=df)
    # min_value = df['covariance'].min()
    # max_value = df['covariance'].max()
    # mapper = LinearColorMapper(palette=colors, low=min_value, high=max_value)
    mapper = LinearColorMapper(palette=colors, low=-1, high=1)
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0))

    bottom, top, left, right = _get_bounds(nlabels=nlabels)

    source = ColumnDataSource(
        dict(top=top, bottom=bottom, left=left, right=right, color_list=color_list,
             taxa_x=df['taxa_x'], taxa_y=df['taxa_y'], covariance=df['covariance']))

    bokeh_tools = ["save, zoom_in, zoom_out, wheel_zoom, box_zoom, crosshair, reset, hover"]

    p = figure(plot_width=width, plot_height=height, x_range=(0, nlabels), y_range=(0, nlabels),
               title=title, title_location='above', x_axis_location="below",
               tools=bokeh_tools, toolbar_location='left')

    p.quad(top="top", bottom="bottom", left="left", right="right", line_color='white',
           color="color_list", source=source)
    p.xaxis.major_label_orientation = pi / 4
    p.yaxis.major_label_orientation = "horizontal"
    p.xaxis.major_label_text_font_size = label_size
    p.yaxis.major_label_text_font_size = label_size
    p.title.text_font_size = title_size
    p.add_layout(color_bar, 'right')
    p.toolbar.autohide = True
    p.xaxis.ticker = [x + 0.5 for x in
                      list(range(0, nlabels))]  ### shift label position to the center
    p.yaxis.ticker = [x + 0.5 for x in list(range(0, nlabels))]
    p.xaxis.major_label_overrides = shifted_labels_dict
    p.yaxis.major_label_overrides = shifted_labels_dict_reversed

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [
        ("taxa_x", "@taxa_x"),
        ("taxa_y", "@taxa_y"),
        ("covariance", "@covariance"),
    ]

    return p


def _make_stats(solution: zarr.hierarchy.Group, labels_dict: dict = None):
    sparsity = flatten_array(solution['modelselect_stats/SP'])
    lambda_path = flatten_array(solution['modelselect_stats/LAMBDA'])
    mu_path = flatten_array(solution['modelselect_stats/MU'])
    ranks = flatten_array(solution['modelselect_stats/RANK'])

    stats = {'sparsity': sparsity, "lambda": lambda_path, 'mu': mu_path, 'rank': ranks}
    df_stats = pd.DataFrame.from_dict(stats, orient='columns')
    source_stats = ColumnDataSource(df_stats)

    columns_stats = [TableColumn(field=col_ix, title=col_ix) for col_ix in df_stats.columns]
    model_selection_stats = DataTable(columns=columns_stats, source=source_stats)

    precision = pd.DataFrame(solution['solution/precision_'])
    pep_stat = pep_metric(matrix=precision)
    best_lambda1 = flatten_array(solution['modelselect_stats/BEST/lambda1'])
    best_mu = flatten_array(solution['modelselect_stats/BEST/mu1'])

    best_stats_dict = {"best lambda": best_lambda1, "best mu": best_mu,
                       "positive edges percentage": pep_stat}
    df_best = pd.DataFrame.from_dict(best_stats_dict, orient='columns')
    source_best = ColumnDataSource(df_best)

    columns_best = [TableColumn(field=col_ix, title=col_ix) for col_ix in df_best.columns]
    best_stats = DataTable(columns=columns_best, source=source_best)

    df = pd.DataFrame(precision.stack(), columns=['covariance']).reset_index()
    df.columns = ["taxa_y", "taxa_x", "covariance"]
    df = df[df['taxa_x'] != df['taxa_y']]  # remove diagonal elements
    df = df.replace({"taxa_x": labels_dict, "taxa_y": labels_dict})
    source_taxa = ColumnDataSource(df)
    columns_taxa = [TableColumn(field=col_ix, title=col_ix) for col_ix in df.columns]
    taxa = DataTable(columns=columns_taxa, source=source_taxa)

    stats_column = column([model_selection_stats, best_stats])
    l1 = row([stats_column, taxa], sizing_mode='fixed')

    return l1


def _solution_plot(solution: zarr.hierarchy.Group, width: int, height: int, label_size: str,
                   clustered: bool = True, n_cov: int = None):
    tabs = []
    labels_dict, labels_dict_reversed = _get_labels(solution=solution, clustered=False)

    # rotate diagonal
    sample_covariance = pd.DataFrame(solution['covariance'])
    precision = pd.DataFrame(solution['solution/precision_'])

    if clustered:
        clust_order = _get_order(sample_covariance, method='average', metric='euclidean')

        S = hierarchical_clustering(sample_covariance, clust_order=clust_order, n_covariates=n_cov)
        Theta = hierarchical_clustering(precision, clust_order=clust_order, n_covariates=n_cov)

        sample_covariance = reset_columns_and_index(S)
        precision = reset_columns_and_index(Theta)

        # new labels order according to the clustering
        labels_dict = {i: labels_dict[key] for i, key in enumerate(clust_order)}
        labels_dict_reversed = {len(labels_dict) - 1 - k: v for k, v in labels_dict.items()}

    p1 = _make_heatmap(data=sample_covariance, title="Sample covariance",
                       width=width, height=height,
                       label_size=label_size, labels_dict=labels_dict,
                       labels_dict_reversed=labels_dict_reversed)
    tab1 = Panel(child=row(p1), title="Sample covariance")
    tabs.append(tab1)

    # due to inversion we multiply the result by -1 to keep the original color scheme
    p2 = _make_heatmap(data=-1 * precision, labels_dict=labels_dict,
                       labels_dict_reversed=labels_dict_reversed,
                       title="Estimated (negative) inverse covariance", width=width, height=height,
                       label_size=label_size)
    tab2 = Panel(child=row(p2), title="Estimated inverse covariance")
    tabs.append(tab2)

    try:
        low_rank = pd.DataFrame(solution['solution/lowrank_'])

        if clustered:
            clust_order = _get_order(sample_covariance, method='average', metric='euclidean')
            L = hierarchical_clustering(low_rank, clust_order=clust_order, n_covariates=n_cov)
            low_rank = reset_columns_and_index(L)

        p3 = _make_heatmap(data=low_rank, labels_dict=labels_dict,
                           labels_dict_reversed=labels_dict_reversed,
                           title="Low-rank", not_low_rank=False, width=width, height=height,
                           label_size=label_size)
        tab3 = Panel(child=row(p3), title="Low-rank")
        tabs.append(tab3)
    except:
        print("NO low-rank solution has been found.")

    p4 = _make_stats(solution=solution, labels_dict=labels_dict)
    tab4 = Panel(child=p4, title="Statistics")
    tabs.append(tab4)

    p = Tabs(tabs=tabs)
    script, div = components(p, INLINE)

    return script, div


def summarize(output_dir: str, solution: zarr.hierarchy.Group,
              width: int = 1500, height: int = 1500, label_size: str = "5pt"):
    J_ENV = jinja2.Environment(
        loader=jinja2.PackageLoader('q2_gglasso._summarize', 'assets')
    )

    template = J_ENV.get_template('index.html')

    script, div = _solution_plot(solution=solution, width=width, height=height,
                                 label_size=label_size)

    output_from_parsed_template = template.render(plot_script=script, plot_div=div)

    with open(os.path.join(output_dir, 'index.html'), 'w') as fh:
        fh.write(output_from_parsed_template)
