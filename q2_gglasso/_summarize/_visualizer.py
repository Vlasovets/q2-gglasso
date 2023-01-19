import os
import zarr
import jinja2
import bisect
import numpy as np
import pandas as pd

from math import pi
from biom.table import Table
from itertools import chain
from q2_gglasso.utils import flatten_array, pep_metric

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


def _get_colors(df: pd.DataFrame(), n_colors: int = 9):
    colors = list(RdBu[n_colors])
    ccorr = np.arange(-1, 1, 1 / (len(colors) / 2))
    color_list = []
    for value in df.covariance.values:
        ind = bisect.bisect_left(ccorr, value)
        color_list.append(colors[ind - 1])
    return color_list, colors


def _get_labels(feature_data: Table, taxonomy: pd.Series):
    taxa = pd.DataFrame(taxonomy.view(pd.Series))

    df = feature_data.to_dataframe()
    df = df.sparse.to_dense()
    ASVs = df.columns.values

    labels = []
    for i in ASVs:
        if i in taxa.index:
            name = taxa.loc[i, "Taxon"]
            labels.append(name)
        else:
            labels.append(i)

    labels_dict = dict(zip(range(len(labels)), labels))
    labels_dict_reversed = dict(zip(range(len(labels)), reversed(labels)))

    return labels_dict, labels_dict_reversed


def _make_heatmap(df: pd.DataFrame(), labels_dict: dict = None, labels_dict_reversed: dict = None, title: str = None,
                  width: int = 1500, height: int = 1500, label_size: str = "5pt", not_low_rank: bool = True):
    nlabels = df.shape[0]
    df = pd.DataFrame(df.stack(), columns=['covariance']).reset_index()
    df.columns = ["taxa_y", "taxa_x", "covariance"]
    if not_low_rank:
        df = df.replace({"taxa_x": labels_dict, "taxa_y": labels_dict})

    color_list, colors = _get_colors(df=df)
    mapper = LinearColorMapper(palette=colors, low=-1, high=1)
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0))

    bottom, top, left, right = _get_bounds(nlabels=nlabels)

    source = ColumnDataSource(dict(top=top, bottom=bottom, left=left, right=right, color_list=color_list,
                                   taxa_x=df['taxa_x'], taxa_y=df['taxa_y'], covariance=df['covariance']))

    bokeh_tools = ["save, zoom_in, zoom_out, wheel_zoom, box_zoom, crosshair, reset, hover"]

    p = figure(plot_width=width, plot_height=height, x_range=(0, nlabels), y_range=(0, nlabels),
               title=title, title_location='above', x_axis_location="below",
               tools=bokeh_tools, toolbar_location='left')

    p.quad(top="top", bottom="bottom", left="left", right="right", line_color='white', color="color_list",
           source=source)
    p.xaxis.major_label_orientation = pi / 4
    p.yaxis.major_label_orientation = "horizontal"
    p.title.text_font_size = "24pt"
    p.add_layout(color_bar, 'right')
    p.toolbar.autohide = True

    p.xaxis.ticker = list(range(0, nlabels))
    p.yaxis.ticker = list(range(0, nlabels))
    if not_low_rank:
        p.xaxis.major_label_overrides = labels_dict
        p.yaxis.major_label_overrides = labels_dict_reversed
    p.xaxis.major_label_text_font_size = label_size
    p.yaxis.major_label_text_font_size = label_size

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

    best_stats_dict = {"best lambda": best_lambda1, "best mu": best_mu, "positive edges percentage": pep_stat}
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


def _solution_plot(solution: zarr.hierarchy.Group, transformed_table: Table, taxonomy: pd.Series,
                   width: int, height: int, label_size: str):
    labels_dict, labels_dict_reversed = _get_labels(feature_data=transformed_table, taxonomy=taxonomy)
    tabs = []

    sample_covariance = pd.DataFrame(solution['covariance']).iloc[::-1]
    p1 = _make_heatmap(df=sample_covariance, labels_dict=labels_dict, labels_dict_reversed=labels_dict_reversed,
                       title="Sample covariance", width=width, height=height, label_size=label_size)
    tab1 = Panel(child=row(p1), title="Sample covariance")
    tabs.append(tab1)

    # due to inversion we multiply the result by -1 to keep the original color scheme
    precision = pd.DataFrame(solution['solution/precision_']).iloc[::-1]
    p2 = _make_heatmap(df=-1 * precision, labels_dict=labels_dict, labels_dict_reversed=labels_dict_reversed,
                       title="Estimated (negative) inverse covariance", width=width, height=height,
                       label_size=label_size)
    tab2 = Panel(child=row(p2), title="Estimated inverse covariance")
    tabs.append(tab2)

    try:
        low_rank = pd.DataFrame(solution['solution/lowrank_']).iloc[::-1]
        p3 = _make_heatmap(df=low_rank, labels_dict=labels_dict, labels_dict_reversed=labels_dict_reversed,
                           title="Low-rank", not_low_rank=False, width=width, height=height, label_size=label_size)
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


def summarize(output_dir: str, solution: zarr.hierarchy.Group, transformed_table: Table, taxonomy: pd.Series,
              width: int = 1500, height: int = 1500, label_size: str = "5pt"):
    J_ENV = jinja2.Environment(
        loader=jinja2.PackageLoader('q2_gglasso._summarize', 'assets')
    )

    template = J_ENV.get_template('index.html')

    script, div = _solution_plot(solution=solution, transformed_table=transformed_table, taxonomy=taxonomy,
                                 width=width, height=height, label_size=label_size)

    output_from_parsed_template = template.render(plot_script=script, plot_div=div)

    with open(os.path.join(output_dir, 'index.html'), 'w') as fh:
        fh.write(output_from_parsed_template)
