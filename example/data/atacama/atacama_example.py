import pandas as pd
import numpy as np
import qiime2 as q2


import matplotlib.pyplot as plt
from scipy import stats

from q2_gglasso.utils import PCA, remove_biom_header


!python setup.py install

!qiime dev refresh-cache

!qiime tools import \
   --type EMPPairedEndSequences \
   --input-path example/data/atacama/emp-paired-end-sequences \
   --output-path example/data/atacama/emp-paired-end-sequences.qza


!qiime demux emp-paired \
  --m-barcodes-file example/data/atacama/sample-metadata.tsv \
  --m-barcodes-column barcode-sequence \
  --p-rev-comp-mapping-barcodes \
  --i-seqs example/data/atacama/emp-paired-end-sequences.qza \
  --o-per-sample-sequences example/data/atacama/demux-full.qza \
  --o-error-correction-details example/data/atacama/demux-details.qza

#subsample to speed up the example
# !qiime demux subsample-paired \
#   --i-sequences example/data/atacama/demux-full.qza \
#   --p-fraction 0.3 \
#   --o-subsampled-sequences example/data/atacama/demux-subsample.qza
#
!qiime demux summarize \
  --i-data example/data/atacama/demux-full.qza \
  --o-visualization example/data/atacama/demux-full.qzv

!qiime tools export \
  --input-path example/data/atacama/demux-full.qzv \
  --output-path example/data/atacama/demux-full/

!qiime demux filter-samples \
  --i-demux example/data/atacama/demux-full.qza \
  --m-metadata-file example/data/atacama/demux-full/per-sample-fastq-counts.tsv \
  --p-where 'CAST([forward sequence count] AS INT) > 100' \
  --o-filtered-demux example/data/atacama/demux.qza

#trimming
!qiime dada2 denoise-paired \
  --i-demultiplexed-seqs example/data/atacama/demux.qza \
  --p-trim-left-f 13 \
  --p-trim-left-r 13 \
  --p-trunc-len-f 150 \
  --p-trunc-len-r 150 \
  --o-table example/data/atacama/table.qza \
  --o-representative-sequences example/data/atacama/rep-seqs.qza \
  --o-denoising-stats example/data/atacama/denoising-stats.qza


!qiime feature-table summarize \
  --i-table example/data/atacama/table.qza \
  --o-visualization example/data/atacama/table.qzv \
  --m-sample-metadata-file example/data/atacama/sample-metadata.tsv


!qiime feature-table tabulate-seqs \
  --i-data example/data/atacama/rep-seqs.qza \
  --o-visualization example/data/atacama/rep-seqs.qzv


!qiime metadata tabulate \
  --m-input-file example/data/atacama/denoising-stats.qza \
  --o-visualization example/data/atacama/denoising-stats.qzv


!qiime feature-table filter-features \
    --i-table example/data/atacama/table.qza \
    --o-filtered-table example/data/atacama/table_100.qza \
    --p-min-frequency 100

!qiime composition add-pseudocount \
                    --i-table example/data/atacama/table_100.qza \
                    --p-pseudocount 1 \
                    --o-composition-table example/data/atacama/table_composition.qza

!qiime gglasso transform-features \
     --p-transformation clr \
     --i-table example/data/atacama/table_composition.qza \
     --o-transformed-table example/data/atacama/table_clr.qza


!qiime tools export \
  --input-path example/data/atacama/table_clr.qza \
  --output-path example/data/atacama/test/table_clr

!qiime gglasso calculate-covariance \
     --p-method scaled \
     --i-table example/data/atacama/table_clr.qza \
     --o-covariance-matrix example/data/atacama/table_corr.qza


!qiime tools export \
  --input-path example/data/atacama/table_corr.qza \
  --output-path example/data/atacama/test/table_corr

!biom convert -i example/data/atacama/test/table_clr/feature-table.biom -o example/data/atacama/test/table_clr/clr_feature-table.tsv --to-tsv

remove_biom_header(file_path="example/data/atacama/test/table_clr/clr_feature-table.tsv")

!qiime gglasso solve-problem \
     --p-lambda1 0.22758 \
     --p-latent True \
     --p-mu1 6.6 \
     --i-covariance-matrix example/data/atacama/table_corr.qza \
     --o-solution example/data/atacama/solution_low.qza \
     --verbose


!qiime gglasso heatmap \
    --i-solution example/data/atacama/solution_low.qza \
    --o-visualization example/data/atacama/heatmap_solution_low.qzv