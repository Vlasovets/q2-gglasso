import os
import pkg_resources
import q2templates
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zarr

TEMPLATES = pkg_resources.resource_filename('q2_gglasso', 'assets')


# TO DO change the palette
heatmap_choices = {
    'color_scheme': {'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                     'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'}
}


def heatmap(output_dir, solution: zarr.hierarchy.Group, color_scheme: str = 'coolwarm') -> None:
    covariance = pd.DataFrame(solution['covariance'])
    precision = pd.DataFrame(solution['solution/precision_'])
    low_rank = pd.DataFrame(solution['solution/lowrank_']) # TO DO version if no low rank

    ## TO DO: test if no matrix is given
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

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
    q2templates.render(index_fp, output_dir, context={})