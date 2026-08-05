# ----------------------------------------------------------------------------
# Copyright (c) 2016-2020, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools import setup, find_packages
import os
import versioneer

# Package meta-data.
NAME = "q2-gglasso"
DESCRIPTION = "QIIME 2 plugin for Single and Multiple Graphical Lasso modeling."
URL = "https://github.com/Vlasovets/q2-gglasso.git"
EMAIL = "oleg.vlasovets@helmholtz-munich.de"
AUTHOR = "Oleg Vlasovets"
REQUIRES_PYTHON = ">=3.10"

# Runtime dependencies. QIIME 2 itself and q2templates are supplied by the
# distribution, so they are deliberately not listed here -- pinning them would
# fight the conda environment. Everything below is a package the distribution
# does NOT ship.
INSTALL_REQUIRES = [
    "gglasso>=0.3.0",
    "numpy>=2.0,<3",
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "numba>=0.60",
    "pandas>=2.2,<3",
    "bokeh>=3.4,<4",
    # zarr 3 removed zarr.hierarchy.Group, which the transformers are annotated
    # with; 2.18.3 is the first release whose __array__ accepts numpy 2's copy=.
    "zarr>=2.18.3,<3",
    "numcodecs>=0.13,<0.16",
    "jinja2>=3.1",
    "biom-format>=2.1.16",
    "seaborn>=0.13",
]

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license="BSD-3-Clause",
    description=DESCRIPTION,
    python_requires=REQUIRES_PYTHON,
    install_requires=INSTALL_REQUIRES,
    extras_require={"dev": ["pytest>=7.0.1", "ruff", "versioneer"]},
    entry_points={"qiime2.plugins": ["q2-gglasso=q2_gglasso.plugin_setup:plugin"]},
    package_data={
        "q2_gglasso": [
            "citations.bib",
            "_summarize/assets/*.html",
            "_summarize/form/*.png",
            # Without this the PCA jinja2 template is missing from a non-editable
            # install and `qiime gglasso pca` fails with TemplateNotFound.
            "_pca/assets/*.html",
        ]
    },
    zip_safe=False,
)
