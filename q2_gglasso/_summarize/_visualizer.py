import os
import pkg_resources
import qiime2
import numpy as np
import zarr
import pandas as pd
import q2templates
from q2_gglasso.utils import flatten_array

TEMPLATES = pkg_resources.resource_filename('q2_gglasso', '_summarize')


def summarize(output_dir, solution: zarr.hierarchy.Group,
              sample_metadata: qiime2.Metadata = None) -> None:

    sparsity = flatten_array(solution['modelselect_stats/SP'])
    lambda_path = flatten_array(solution['modelselect_stats/LAMBDA'])
    mu_path = flatten_array(solution['modelselect_stats/MU'])
    ranks = flatten_array(solution['modelselect_stats/RANK'])

    stats = {'sparsity': sparsity, "lambda_path": lambda_path,
             'mu_path': mu_path, 'ranks': ranks}

    stats_df = pd.DataFrame.from_dict(stats)

    stats_df.to_csv(os.path.join(output_dir, 'stats.csv'))

    stats_table = q2templates.df_to_html(stats_df)
    context = {'stats_table': stats_table}

    stats_table_template = os.path.join(TEMPLATES, 'assets', 'index.html')
    index = os.path.join(TEMPLATES, 'assets', 'index.html')

    templates = [index, stats_table_template]
    q2templates.render(templates, output_dir, context=context)