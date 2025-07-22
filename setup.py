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
URL = "https://github.com/Vlasovets/q2-gglasso"
EMAIL = "otorrent@mail.ru"
AUTHOR = "Oleg Vlasovets"
REQUIRES_PYTHON = ">=3.6.0"

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
    entry_points={"qiime2.plugins": ["q2-gglasso=q2_gglasso.plugin_setup:plugin"]},
    package_data={
        "q2_gglasso": [
            "citations.bib",
            "_summarize/assets/*.html",
            "_summarize/form/*.png",
        ]
    },
    zip_safe=False,
)
