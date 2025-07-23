# q2-gglasso
[![PyPI license](https://img.shields.io/pypi/l/gglasso.svg)](https://pypi.python.org/pypi/gglasso/)
[![Python version](https://img.shields.io/badge/python-%3E3.6-blue)](https://www.python.org/)
[![Documentation Status](https://readthedocs.org/projects/gglasso/badge/?version=latest)](http://gglasso.readthedocs.io/?badge=latest)
[![CI](https://github.com/Vlasovets/q2-gglasso/actions/workflows/ci.yml/badge.svg?branch=dev&event=push)](https://github.com/Vlasovets/q2-gglasso/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*A [QIIME 2](https://qiime2.org) plugin for solving **General Graphical Lasso (GGLasso)** problems with microbiome data including:*

- Single Graphical Lasso (SGL)
- Adaptive SGL
- SGL with latent variables
- Multiple Graphical Lasso (MGL)
- MGL with latent variables
- GGL in the nonconforming case

📚 [Documentation](https://gglasso.readthedocs.io/en/latest/) |
📂 [Tutorial & Examples](https://github.com/Vlasovets/atacama-soil-microbiome-tutorial)

---

## Installation

First, make sure you have GGLasso is installed:

```
conda install -q gglasso
```

OR

```
pip install gglasso
```

Next to install the plugin:

```
pip install git+https://github.com/bio-datascience/q2-gglasso.git
qiime dev refresh-cache
```

---

## Usage Tutorial

A complete tutorial on using `q2-gglasso` for microbiome data analysis — including preprocessing, CLR transformation, model fitting, and visualization — is available at:

👉 **[Atacama Soil Microbiome Tutorial](https://github.com/Vlasovets/atacama-soil-microbiome-tutorial)**

This tutorial includes:

- Input formatting and CLR transformation
- Applications of SGL, adaptive SGL, and SGL with latent variables
- Network visualizations of ASV associations
- Covariate analysis and interpretation

---

## Citation

If you use `q2-gglasso`, please cite:

> Schaipp, F., Vlasovets, O., & Müller, C. L. (2021). **GGLasso – a Python package for General Graphical Lasso computation**. *Journal of Open Source Software, 6*(68), 3865. [https://doi.org/10.21105/joss.03865](https://doi.org/10.21105/joss.03865)

## Related Projects

- [`gglasso`](https://github.com/Vlasovets/gglasso): Python solvers for graphical lasso problems
- [`atacama-soil-microbiome-tutorial`](https://github.com/Vlasovets/atacama-soil-microbiome-tutorial): Full tutorial and example analyses

---

## License

BSD 3-Clause License. See [LICENSE](./LICENSE) for details.