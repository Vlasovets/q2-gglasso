import os.path
import pkg_resources

import q2templates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qiime2


TEMPLATES = pkg_resources.resource_filename('q2_feature_table._heatmap',
                                            'assets')

heatmap_choices = {
    'color_scheme': {'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'}
}
