import os
import zarr
import jinja2
import bisect
import numpy as np
import pandas as pd

from math import pi
from itertools import chain
from q2_gglasso.utils import flatten_array, pep_metric

from bokeh.plotting import figure
from bokeh.models import HoverTool, Panel, Tabs, ColorBar, LinearColorMapper
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.palettes import RdBu
from bokeh.layouts import row


def _get_bounds(nlabels: int):
    """
    Get plotting grid bounds.
    Parameters
    ----------
    nlabels: int
        Number of labels
    Returns
    -------
    bottom, top, left, right: list
        Grid of dimensions for plotting.
    """
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


def _make_heatmap(df: pd.DataFrame(), title: str = None, width: int = 1500, height: int = 1500):
    df = pd.DataFrame(df.stack(), columns=['covariance']).reset_index()
    df.columns = ["taxa_x", "taxa_y", "covariance"]

    labels = df.taxa_x.unique()
    nlabels = len(labels)
    color_list, colors = _get_colors(df=df)
    mapper = LinearColorMapper(palette=colors, low=-1, high=1)
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0))

    bottom, top, left, right = _get_bounds(nlabels=nlabels)

    source = ColumnDataSource(dict(top=top, bottom=bottom, left=left, right=right, color_list=color_list,
                                   taxa_x=df['taxa_x'], taxa_y=df['taxa_y'], covariance=df['covariance']))

    bokeh_tools = ["save, zoom_in, zoom_out, wheel_zoom, box_zoom, crosshair, reset, hover"]

    p = figure(plot_width=width, plot_height=height, x_range=(0, nlabels), y_range=(0, nlabels),
               title=title, title_location='above', x_axis_location="above",
               tools=bokeh_tools, toolbar_location='left')

    p.quad(top="top", bottom="bottom", left="left", right="right", line_color='white',
           color="color_list", source=source)
    p.xaxis.major_label_orientation = pi / 4
    p.yaxis.major_label_orientation = "horizontal"
    p.title.text_font_size = '24pt'
    p.add_layout(color_bar, 'right')
    p.toolbar.autohide = True

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [
        ("taxa_x", "@taxa_x"),
        ("taxa_y", "@taxa_y"),
        ("covariance", "@covariance"),
    ]

    return p


def _make_stats(solution: zarr.hierarchy.Group):
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

    l1 = row([model_selection_stats, best_stats], sizing_mode='fixed')

    return l1


def _solution_plot(solution: zarr.hierarchy.Group):
    covariance = pd.DataFrame(solution['covariance']).iloc[::-1]
    p1 = _make_heatmap(df=covariance, title="Covariance")
    tab1 = Panel(child=row(p1), title="Covariance")

    precision = pd.DataFrame(solution['solution/precision_']).iloc[::-1]
    p2 = _make_heatmap(df=precision, title="Precision")
    tab2 = Panel(child=row(p2), title="Precision")

    low_rank = pd.DataFrame(solution['solution/lowrank_']).iloc[::-1]
    p3 = _make_heatmap(df=low_rank, title="Low-rank")
    tab3 = Panel(child=row(p3), title="Low-rank")

    p4 = _make_stats(solution=solution)
    tab4 = Panel(child=p4, title="Statistics")

    tabs = [tab1, tab2, tab3, tab4]
    p = Tabs(tabs=tabs)
    script, div = components(p, INLINE)

    return script, div


def summarize(output_dir: str, solution: zarr.hierarchy.Group):
    J_ENV = jinja2.Environment(
        loader=jinja2.PackageLoader('q2_gglasso._summarize', 'assets')
    )

    template = J_ENV.get_template('index.html')
    print(template.render())

    script, div = _solution_plot(solution=solution)

    output_from_parsed_template = template.render(plot_script=script, plot_div=div)

    with open(os.path.join(output_dir, 'index.html'), 'w') as fh:
        fh.write(output_from_parsed_template)
