import os
import pkg_resources
import numpy as np
import zarr
import pandas as pd
import q2templates
from q2_gglasso.utils import flatten_array

TEMPLATES = pkg_resources.resource_filename('q2_gglasso._summarize', '_summarize')


def summarize(output_dir, solution: zarr.hierarchy.Group) -> None:

    sparsity = flatten_array(solution['modelselect_stats/SP'])
    lambda_path = flatten_array(solution['modelselect_stats/LAMBDA'])
    mu_path = flatten_array(solution['modelselect_stats/MU'])
    ranks = flatten_array(solution['modelselect_stats/RANK'])

    stats = {'sparsity': sparsity, "lambda_path": lambda_path,
             'mu_path': mu_path, 'ranks': ranks}

    stats_df = pd.DataFrame.from_dict(stats)

    stats_table = q2templates.df_to_html(stats_df)

    context = {'feature_frequencies_table': stats_table}

    index_fp = os.path.join(TEMPLATES, 'index.html')
    q2templates.render(index_fp, output_dir, context=context)