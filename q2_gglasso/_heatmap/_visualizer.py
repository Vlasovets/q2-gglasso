import os
import pkg_resources
import q2templates
import pandas as pd
import seaborn as sns

TEMPLATES = pkg_resources.resource_filename('q2_gglasso._heatmap', 'assets')

# TO DO change the palette
heatmap_choices = {
    'color_scheme': {'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                     'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'}
}


def heatmap(output_dir, covariance_matrix: pd.DataFrame, normalize: bool = True) -> None:
    heatmap_plot = sns.heatmap(covariance_matrix, cmap="coolwarm", vmin=-0.5, vmax=0.5, linewidth=0.5,
                               xticklabels=[], yticklabels=[], square=True, cbar=False)
    fig = heatmap_plot.get_figure()

    for ext in ['png', 'svg']:
        img_fp = os.path.join(output_dir, 'gglasso-heatmap.{0}'.format(ext))
        fig.savefig(img_fp)

    index_fp = os.path.join(TEMPLATES, 'index.html')
    q2templates.render(index_fp, output_dir, context={'normalize': normalize})
