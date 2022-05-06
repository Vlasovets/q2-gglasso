import pandas as pd
import numpy as np
import qiime2 as q2

import matplotlib.pyplot as plt
from scipy import stats

from q2_gglasso._func import PCA, remove_biom_header

!python setup.py install

!qiime dev refresh-cache

!qiime feature-table filter-features \
    --i-table example/data/88soils.biom.qza \
    --o-filtered-table example/data/88soils_filt100.biom.qza \
    --p-min-frequency 100

!qiime composition add-pseudocount \
                    --i-table example/data/88soils_filt100.biom.qza \
                    --p-pseudocount 1 \
                    --o-composition-table example/data/88soils_composition.biom.qza

!qiime gglasso transform-features \
     --p-transformation clr \
     --i-table example/data/88soils_composition.biom.qza \
     --o-transformed-table example/data/88soils_clr.biom.qza

!qiime tools export \
  --input-path example/data/88soils_clr.biom.qza \
  --output-path example/data/test_soil/88soils_clr

!biom convert -i example/data/test_soil/88soils_clr/feature-table.biom -o example/data/test_soil/88soils_clr/clr_feature-table.tsv --to-tsv

#remove biom header from the files
remove_biom_header(file_path="example/data/test_soil/88soils_clr/clr_feature-table.tsv")

#covariance
!qiime gglasso calculate-covariance \
     --p-method unscaled \
     --i-table example/data/88soils_clr.biom.qza \
     --o-covariance-matrix example/data/88soils_covariance.qza

# even there is no low rank solution we have to specify the output
!qiime gglasso solve-problem \
     --p-lambda1 0.05 \
     --i-covariance-matrix example/data/88soils_covariance.qza \
     --o-solution example/data/88soils_solution.qza


# !qiime gglasso heatmap \
#     --i-covariance example/data/88soils_covariance.qza \
#     --i-inv-covariance example/data/88soils_inverse.qza \
#     --i-low-rank example/data/88soils_low.qza \
#     --o-visualization example/data/88soils_heatmap_new.qzv

#correlation
!qiime gglasso calculate-covariance \
     --p-method scaled \
     --i-table example/data/88soils_clr.biom.qza \
     --o-covariance-matrix example/data/88soils_corr.qza

# optimal lambda 0.22758459260747887, mu 6.6
!qiime gglasso solve-problem \
     --p-lambda1 0.22758 \
     --p-latent True \
     --p-mu1 6.6 \
     --i-covariance-matrix example/data/88soils_corr.qza \
     --o-solution example/data/88soils_solution_low.qza

!qiime gglasso heatmap \
    --i-solution example/data/88soils_solution_low.qza \
    --o-visualization example/data/88soils_heatmap_new.qzv


mapping = pd.read_table('example/data/88soils_metadata.txt', index_col=0)
mapping['ph_rounded'] = mapping.ph.apply(int)

df = pd.read_csv(str("example/data/test_soil/88soils_clr/clr_feature-table.tsv"), index_col=0, sep='\t')
ph = mapping['ph'].reindex(df.index)
temperature = mapping["annual_season_temp"].reindex(ph.index)

depth = df.sum(axis=1)

L = pd.read_csv(str("example/data/test_soil/88soils_low/pairwise_comparisons.tsv"), index_col=0, sep='\t')


proj, loadings, eigv = PCA(df, L, inverse=True)
r = np.linalg.matrix_rank(L)


fig, ax = plt.subplots(1,1)
im = ax.scatter(proj[:,0], ph, c = depth, cmap = plt.cm.Blues, vmin = 0)
cbar = fig.colorbar(im)
cbar.set_label("Sampling depth")
ax.set_xlabel(f"PCA component 1 with eigenvalue {eigv[0]}")
ax.set_ylabel("pH")
plt.savefig('example/data/ph.png')

print("Spearman correlation between pH and 1st component: {0}, p-value: {1}".format(stats.spearmanr(ph, proj[:,0])[0],
                                                                              stats.spearmanr(ph, proj[:,0])[1]))


fig, ax = plt.subplots(1,1)
im = ax.scatter(proj[:,1], temperature, c = depth, cmap = plt.cm.Blues, vmin = 0)
cbar = fig.colorbar(im)
cbar.set_label("Sampling depth")
ax.set_xlabel(f"PCA component 2 with eigenvalue {eigv[1]}")
ax.set_ylabel("Temperature")
plt.savefig('example/data/temp.png')


print("Spearman correlation between temperature and 2nd component: {0}, p-value: {1}".format(stats.spearmanr(temperature, proj[:,1])[0],
                                                                              stats.spearmanr(temperature, proj[:,1])[1]))
