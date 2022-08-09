import qiime2
import os
import pkg_resources
import q2templates
import matplotlib.pyplot as plt
import zarr
from biom.table import Table
from q2_gglasso.utils import correlated_PC

TEMPLATES = pkg_resources.resource_filename('q2_gglasso._pca', 'assets')


def pca(output_dir, table: Table, solution: zarr.hierarchy.Group, corr_bound: float = 0.7, alpha: float = 0.05,
        sample_metadata: qiime2.Metadata = None) -> None:

    df = table.to_dataframe()

    numeric_md_cols = sample_metadata.filter_columns(column_type='numeric')
    md = numeric_md_cols.to_dataframe()
    md = md.reindex(df.index)

    L = solution['solution/lowrank_']

    proj_dict = correlated_PC(data=df, metadata=md, low_rank=L, corr_bound=corr_bound, alpha=alpha)

    fig, ax = plt.subplots(len(proj_dict.keys()), 1, figsize=(5, 50))

    i = 0
    for col, proj in proj_dict.items():
        for key in proj.keys():

            plot_df = proj['data']
            textstr = '\n'.join((r'$\rho=%.2f$' % (proj['rho'],),
                                 r'$\mathrm{p_{value}}=%.2f$' % (proj['p_value'],),
                                 r'$eig_{value}=%.3f$' % (proj['eigenvalue'],)))

            if "PC" in key:
                im = ax[i].scatter(proj[key], plot_df[col], c=plot_df['sequencing depth'], cmap=plt.cm.Blues, vmin=0)
                ax[i].set_xlabel(key)
                ax[i].set_ylabel(col)
                cbar = plt.colorbar(im, ax=ax[i])
                cbar.set_label("Sampling depth")

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
                ax[i].text(0.05, 0.95, textstr, transform=ax[i].transAxes, fontsize=7, verticalalignment='top',
                           bbox=props)

        i += 1

    for ext in ['png', 'svg']:
        img_fp = os.path.join(output_dir, 'q2-gglasso-pca.{0}'.format(ext))
        plt.savefig(img_fp)

    index_fp = os.path.join(TEMPLATES, 'index.html')
    q2templates.render(index_fp, output_dir, context={})
