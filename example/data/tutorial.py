!qiime feature-table filter-features \
  --i-table table.qza \
  --p-min-samples 20 \
  --o-filtered-table filtered-table.qza


!qiime taxa collapse --i-table table.qza \
  --i-taxonomy taxonomy.qza \
  --p-level 6 \
  --o-collapsed-table genus_table.qza


