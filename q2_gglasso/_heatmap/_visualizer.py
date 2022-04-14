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



# from gglasso.helper.data_generation import generate_precision_matrix, group_power_network, sample_covariance_matrix
# from gglasso.problem import glasso_problem
#
# p = 20
# N = 1000
#
# Sigma, Theta = generate_precision_matrix(p=p, M=1, style='erdos', prob=0.1, seed=1234)
#
# S, sample = sample_covariance_matrix(Sigma, N)
#
# P = glasso_problem(S, N, reg_params = {'lambda1': 0.05}, latent = False, do_scaling = False)
# print(P)
#
# P.solve()
#
# heatmap_plot = sns.heatmap(P.solution.precision_, cmap="coolwarm", vmin=-0.5,
#             vmax=0.5, linewidth=0.5, xticklabels = [], yticklabels = [],
#             square=True, cbar=False)
#
# fig = heatmap_plot.get_figure()
#
# for ext in ['png', 'svg']:
#     img_fp = os.path.join(os.path.join(os.getcwd(), "example/data/"), 'gglasso-heatmap.{0}'.format(ext))
#     fig.savefig(img_fp)
#
# os.path.join(os.getcwd(), "example/data/")

'gglasso-heatmap.{0}'.format('png')

def heatmap(output_dir, covariance_matrix: pd.DataFrame, normalize: bool = True) -> None:

    heatmap_plot = sns.heatmap(covariance_matrix, cmap="coolwarm", vmin=-0.5, vmax=0.5, linewidth=0.5,
                               xticklabels=[], yticklabels=[], square=True, cbar=False)
    fig = heatmap_plot.get_figure()

    for ext in ['png', 'svg']:
        img_fp = os.path.join(output_dir, 'gglasso-heatmap.{0}'.format(ext))
        fig.savefig(img_fp)

    index_fp = os.path.join(TEMPLATES, 'index.html')
    q2templates.render(index_fp, output_dir, context={'normalize': normalize})
