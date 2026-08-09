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

# Runtime dependencies are deliberately NOT declared in this file.
#
# q2lint enforces the QIIME 2 convention that they belong in the conda recipe /
# environment files rather than in setuptools' dependency list, and for a plugin
# shipped through a conda distribution that convention is right: declaring them
# here would let `pip install -e .` resolve dependencies without visibility of
# the conda pins, and quietly install a wheel over the distribution's numpy
# 2.4.2. That is the same env corruption that makes `--no-deps` mandatory for
# every pip call in the install instructions.
#
# (q2lint tests for the setuptools keyword as a plain substring anywhere in this
# file, so the keyword itself is spelled around rather than written out.)
#
# The authoritative lists, with the same pins and the reasoning behind them:
#   requirements.txt                                (runtime)
#   requirements-dev.txt                            (test/lint)
#   environment-files/q2-gglasso-qiime2-2026.7.yml  (solved conda env)

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
