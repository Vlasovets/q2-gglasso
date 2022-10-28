import os
import pandas as pd
import zarr
import pandas as pd
import jinja2
import bisect
import pkg_resources
import numpy as np
from math import pi
from itertools import chain
from q2_gglasso.utils import flatten_array, pep_metric

from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, Panel, Tabs, ColorBar, LinearColorMapper, BasicTicker, PrintfTickFormatter
from bokeh.models import ColumnDataSource, TableColumn, DataTable, Quad
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.palettes import RdBu
from bokeh.io import export_svg
from bokeh.layouts import row, column, layout


def summarize(output_dir: str, solution: zarr.hierarchy.Group):
    # stats, pep_stat, best_lambda1, best_mu = _make_stats(solution)
    #
    # df = pd.DataFrame.from_dict(stats, orient='columns')
    #
    # html = df.to_html(index=False)

    # table1 = ('Graphical lasso model selection statistics with 'PEP={0}, best lambda={1}, best mu={2}.'.format(pep_stat, best_lambda1, best_mu))

    # index_fp = _make_heatmap(output_dir, solution, title=title)

    J_ENV = jinja2.Environment(
        loader=jinja2.PackageLoader('q2_gglasso._summarize', 'assets')
    )

    template = J_ENV.get_template('index.html')
    print(template.render())

    script, div = _solution_plot(solution=solution)

    # output_from_parsed_template = template.render(stats=html, plot_script=script, plot_div=div)
    output_from_parsed_template = template.render(plot_script=script, plot_div=div)

    with open(os.path.join(output_dir, 'index.html'), 'w') as fh:
        fh.write(output_from_parsed_template)


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
                                   taxa_x=df['taxa_x'],
                                   taxa_y=df['taxa_y'],
                                   covariance=df['covariance']
                                   ))

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


def _solution_plot(solution: zarr.hierarchy.Group):
    covariance = pd.DataFrame(solution['covariance']).iloc[::-1]
    p1 = _make_heatmap(df=covariance, title="Covariance")
    tab1 = Panel(child=p1, title="Covariance")

    precision = pd.DataFrame(solution['solution/precision_']).iloc[::-1]
    p2 = _make_heatmap(df=precision, title="Precision")
    tab2 = Panel(child=p2, title="Precision")

    low_rank = pd.DataFrame(solution['solution/lowrank_']).iloc[::-1]
    p3 = _make_heatmap(df=low_rank, title="Low-rank")
    tab3 = Panel(child=p3, title="Low-rank")

    # for plot in [p1, p2, p3]:
    #     export_svg(plot, filename="{0}_heatmap.svg".format(plot.title.text))

    # p4 = _make_stats(solution=solution)
    # tab1_layout = layout([p4])
    # tab4 = Panel(child=tab1_layout, title="Statistics")

    tabs = [tab1, tab2, tab3]
    p = Tabs(tabs=tabs)
    script, div = components(p, INLINE)

    return script, div


solution = zarr.load("data/atacama_low/problem.zip")
#
# def _make_stats(solution: zarr.hierarchy.Group):
#     sparsity = flatten_array(solution['modelselect_stats/SP'])
#     lambda_path = flatten_array(solution['modelselect_stats/LAMBDA'])
#     mu_path = flatten_array(solution['modelselect_stats/MU'])
#     ranks = flatten_array(solution['modelselect_stats/RANK'])
#
#     precision = pd.DataFrame(solution['solution/precision_'])
#
#     total_edges = np.count_nonzero(precision) / 2
#     positive_edges = np.sum(precision > 0, axis=0)
#     total_positives = np.sum(positive_edges) / 2
#     pep_stat = np.round(total_positives / total_edges, 2)
#
#     best_lambda1 = flatten_array(solution['modelselect_stats/BEST/lambda1']).item()
#     best_mu = flatten_array(solution['modelselect_stats/BEST/mu1']).item()
#
#     stats = {'sparsity': sparsity, "lambda": lambda_path, 'mu': mu_path, 'rank': ranks}
#
#     df = pd.DataFrame.from_dict(stats, orient='columns')
#     source = ColumnDataSource(df)
#
#     columns = [
#         TableColumn(field='sparsity', title='sparsity'),
#         TableColumn(field='sambda', title='lambda'),
#         TableColumn(field='mu', title='mu'),
#         TableColumn(field='rank', title='rank'),
#     ]
#
#     myTable = DataTable(source=source, columns=columns)
#
#     tables_layout = column()
#     for col in columns:
#         tables_layout.children.append(myTable)
#
#     return tables_layout


from bokeh.io import show


def _make_stats(solution: zarr.hierarchy.Group):
    sparsity = flatten_array(solution['modelselect_stats/SP'])
    lambda_path = flatten_array(solution['modelselect_stats/LAMBDA'])
    mu_path = flatten_array(solution['modelselect_stats/MU'])
    ranks = flatten_array(solution['modelselect_stats/RANK'])

    precision = pd.DataFrame(solution['solution/precision_'])
    pep_stat = pep_metric(matrix=precision)

    best_lambda1 = flatten_array(solution['modelselect_stats/BEST/lambda1']).item()
    best_mu = flatten_array(solution['modelselect_stats/BEST/mu1']).item()

    stats = {'sparsity': sparsity, "lambda": lambda_path, 'mu': mu_path, 'rank': ranks}

    df = pd.DataFrame.from_dict(stats, orient='columns')
    source = ColumnDataSource(df)

    columns = [TableColumn(field=col_ix, title=col_ix) for col_ix in df.columns]
    stats_table = DataTable(columns=columns, source=source)

    J_ENV = jinja2.Environment(
        loader=jinja2.PackageLoader('q2_gglasso._summarize', 'assets')
    )

    template = J_ENV.get_template('index.html')
    print(template.render())
    script, div = components(stats_table, INLINE)

    output_from_parsed_template = template.render(plot_script=script, plot_div=div)

    # p4 = _make_stats(solution=solution)
    # tab1_layout = layout([p4])
    # tab4 = Panel(child=tab1_layout, title="Statistics")

    print(output_from_parsed_template)
    # # to save the results
    with open("/opt/project/my_new_file.html", "w") as fh:
        fh.write(output_from_parsed_template)

    return stats, pep_stat, best_lambda1, best_mu

#
# def _make_heatmap(output_dir: str, solution: zarr.hierarchy.Group, color_scheme: str = 'coolwarm', title: str = None):
#     covariance = pd.DataFrame(solution['covariance'])
#     precision = pd.DataFrame(solution['solution/precision_'])
#     low_rank = pd.DataFrame(solution['solution/lowrank_'])  # TO DO version if no low rank
#
#     total_edges = np.count_nonzero(precision) / 2
#     positive_edges = np.sum(precision > 0, axis=0)
#     total_positives = np.sum(positive_edges) / 2
#     pep_stat = np.round(total_positives / total_edges, 2)
#
#     best_lambda1 = flatten_array(solution['modelselect_stats/BEST/lambda1']).item()
#     best_mu = flatten_array(solution['modelselect_stats/BEST/mu1']).item()
#
#     ## TO DO: test if no matrix is given
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
#     fig.suptitle(title + " (lambda_opt: {0}, mu_opt: {1}, PEP: {2}).".format(best_lambda1, best_mu, pep_stat), fontsize=26)
#
#     ax1.get_shared_y_axes().join(ax2, ax3)
#     g1 = sns.heatmap(covariance, cmap=color_scheme, cbar=False, ax=ax1)
#     g1.set_ylabel('')
#     g1.set_xlabel('Covariance')
#
#     g2 = sns.heatmap(precision, cmap=color_scheme, cbar=False, ax=ax2)
#     g2.set_ylabel('')
#     g2.set_xlabel('Inverse covariance')
#     g2.set_yticks([])
#
#     g3 = sns.heatmap(low_rank, cmap=color_scheme, ax=ax3, cbar=False)
#     g3.set_ylabel('')
#     g3.set_xlabel('Low-rank solution')
#     g3.set_yticks([])
#
#     for ax in [g1, g2, g3]:
#         tl = ax.get_xticklabels()
#         ax.set_xticklabels(tl, rotation=90)
#         tly = ax.get_yticklabels()
#         ax.set_yticklabels(tly, rotation=0)
#
#     for ext in ['png', 'svg']:
#         img_fp = os.path.join(output_dir, 'q2-gglasso-heatmap.{0}'.format(ext))
#         plt.savefig(img_fp)
#
#     index_fp = os.path.join(TEMPLATES, 'assets/index.html')
#
#     return index_fp
