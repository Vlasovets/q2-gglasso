import os
import zarr
import pandas as pd
import jinja2
from q2_gglasso.utils import flatten_array


def summarize(output_dir: str, solution: zarr.hierarchy.Group):
    sparsity = flatten_array(solution['modelselect_stats/SP'])
    lambda_path = flatten_array(solution['modelselect_stats/LAMBDA'])
    mu_path = flatten_array(solution['modelselect_stats/MU'])
    ranks = flatten_array(solution['modelselect_stats/RANK'])

    stats = {'sparsity': sparsity, "lambda": lambda_path,
             'mu': mu_path, 'rank': ranks}

    df = pd.DataFrame.from_dict(stats, orient='columns')

    html = df.to_html(index=False)

    table1 = (f'Graphical lasso model selection statistics.')

    J_ENV = jinja2.Environment(
        loader=jinja2.PackageLoader('q2_gglasso._summarize', 'assets')
    )

    index = J_ENV.get_template('index.html')

    with open(os.path.join(output_dir, 'index.html'), 'w') as fh:
        fh.write(index.render(stats=html, table1=table1))


# Theta = solution['solution/precision_']
# total_edges = np.count_nonzero(Theta) / 2
# total_positives = np.sum(np.sum(Theta > 0, axis=0)) / 2
# pep_stat = total_positives / total_edges