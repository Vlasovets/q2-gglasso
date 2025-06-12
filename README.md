# q2-gglasso
[![PyPI license](https://img.shields.io/pypi/l/gglasso.svg)](https://pypi.python.org/pypi/gglasso/)
[![Python version](https://img.shields.io/badge/python-%3E3.6-blue)](https://www.python.org/)
[![Documentation Status](https://readthedocs.org/projects/gglasso/badge/?version=latest)](http://gglasso.readthedocs.io/?badge=latest)
[![Tests](https://github.com/Vlasovets/q2-gglasso/actions/workflows/ci.yml/badge.svg?branch=dev&event=push)](https://github.com/Vlasovets/q2-gglasso/actions/workflows/ci.yml)

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

Instructions for installing `q2-gglasso` via Conda are coming soon.
In the meantime, please refer to the [official documentation](https://gglasso.readthedocs.io/en/latest/) for setup instructions and requirements.

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
- [`q2-gglasso`](https://github.com/Vlasovets/q2-gglasso): QIIME 2 plugin (this repository)
- [`atacama-soil-microbiome-tutorial`](https://github.com/Vlasovets/atacama-soil-microbiome-tutorial): Full tutorial and example analyses

---

## License

MIT License. See [LICENSE](./LICENSE) for details.