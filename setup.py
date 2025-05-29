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
NAME = 'q2-gglasso'
DESCRIPTION = 'Algorithms for Single and Multiple Graphical Lasso problems.'
URL = 'https://github.com/Vlasovets/q2-gglasso'
EMAIL = 'otorrent@mail.ru'
AUTHOR = 'Oleg Vlasovets'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = "0.0.1"

here = os.path.abspath(os.path.dirname("__file__"))

# Read requirements from file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="q2-gglasso",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    author="Oleg Vlasovets",
    author_email="otorrent@mail.ru",
    url="https://qiime2.org",
    license="BSD-3-Clause",
    description="Package for Multiple Graphical Lasso problem",
    entry_points={
        "qiime2.plugins":
            ["q2-gglasso=q2_gglasso.plugin_setup:plugin"]
    },
    package_data={'q2_gglasso': ['citations.bib', '_summarize/assets/*.html', '_summarize/form/*.png']},
    zip_safe=False,
    install_requires=requirements,
)
