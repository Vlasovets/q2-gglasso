import os
import zarr
import pandas as pd
import jinja2
import seaborn as sns
import pkg_resources
import numpy as np
import q2templates
import matplotlib.pyplot as plt
from q2_gglasso.utils import flatten_array

TEMPLATES = pkg_resources.resource_filename('q2_gglasso._heatmap', 'assets')

def summarize(output_dir: str, solution: zarr.hierarchy.Group, title: str):

    stats = _make_stats(solution)

    df = pd.DataFrame.from_dict(stats, orient='columns')

    html = df.to_html(index=False)

    table1 = (f'Graphical lasso model selection statistics.')

    index_fp = _make_heatmap(output_dir, solution, title=title)

    J_ENV = jinja2.Environment(
        loader=jinja2.PackageLoader('q2_gglasso._summarize', 'assets')
    )

    index = J_ENV.get_template('index.html')

    with open(os.path.join(output_dir, 'index.html'), 'w') as fh:
        fh.write(index.render(stats=html, table1=table1, figure1=index_fp))


solution = zarr.load("data/atacama_low/problem.zip")
color_scheme = 'coolwarm'

def _make_heatmap(output_dir: str, solution: zarr.hierarchy.Group, color_scheme: str = 'coolwarm', title: str = None):
    covariance = pd.DataFrame(solution['covariance'])
    precision = pd.DataFrame(solution['solution/precision_'])
    low_rank = pd.DataFrame(solution['solution/lowrank_'])  # TO DO version if no low rank

    total_edges = np.count_nonzero(precision) / 2
    positive_edges = np.sum(precision > 0, axis=0)
    total_positives = np.sum(positive_edges) / 2
    pep_stat = np.round(total_positives / total_edges, 2)

    best_lambda1 = flatten_array(solution['modelselect_stats/BEST/lambda1']).item()
    best_mu = flatten_array(solution['modelselect_stats/BEST/mu1']).item()

    ## TO DO: test if no matrix is given
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(title + " (lambda_opt: {0}, mu_opt: {1}, PEP: {2}).".format(best_lambda1, best_mu, pep_stat), fontsize=26)

    ax1.get_shared_y_axes().join(ax2, ax3)
    g1 = sns.heatmap(covariance, cmap=color_scheme, cbar=False, ax=ax1)
    g1.set_ylabel('')
    g1.set_xlabel('Covariance')

    g2 = sns.heatmap(precision, cmap=color_scheme, cbar=False, ax=ax2)
    g2.set_ylabel('')
    g2.set_xlabel('Inverse covariance')
    g2.set_yticks([])

    g3 = sns.heatmap(low_rank, cmap=color_scheme, ax=ax3, cbar=False)
    g3.set_ylabel('')
    g3.set_xlabel('Low-rank solution')
    g3.set_yticks([])

    for ax in [g1, g2, g3]:
        tl = ax.get_xticklabels()
        ax.set_xticklabels(tl, rotation=90)
        tly = ax.get_yticklabels()
        ax.set_yticklabels(tly, rotation=0)

    for ext in ['png', 'svg']:
        img_fp = os.path.join(output_dir, 'q2-gglasso-heatmap.{0}'.format(ext))
        plt.savefig(img_fp)

    index_fp = os.path.join(TEMPLATES, 'index.html')

    return index_fp


def _make_stats(solution: zarr.hierarchy.Group):

    sparsity = flatten_array(solution['modelselect_stats/SP'])
    lambda_path = flatten_array(solution['modelselect_stats/LAMBDA'])
    mu_path = flatten_array(solution['modelselect_stats/MU'])
    ranks = flatten_array(solution['modelselect_stats/RANK'])

    stats = {'sparsity': sparsity, "lambda": lambda_path, 'mu': mu_path, 'rank': ranks}

    return stats



