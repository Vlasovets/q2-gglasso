# q2-gglasso
*A [QIIME 2](https://qiime2.org) plugin for solving **General Graphical Lasso (GGLasso)** problems with microbiome data including:*

- Single Graphical Lasso (SGL)
- Adaptive SGL
- SGL with latent variables
- Multiple Graphical Lasso (MGL)
- MGL with latent variables
- GGL in the nonconforming case

[![PyPI license](https://img.shields.io/pypi/l/gglasso.svg)](https://pypi.python.org/pypi/gglasso/)
[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)
[![Documentation Status](https://readthedocs.org/projects/gglasso/badge/?version=latest)](http://gglasso.readthedocs.io/?badge=latest)

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

> Vlasovets, O. et al. *Computational methods for microbiome graphical modeling with applications to allergy and environmental exposure*. (Forthcoming)

---

## Related Projects

- [`gglasso`](https://github.com/Vlasovets/gglasso): Python solvers for graphical lasso problems
- [`q2-gglasso`](https://github.com/Vlasovets/q2-gglasso): QIIME 2 plugin (this repository)
- [`atacama-soil-microbiome-tutorial`](https://github.com/Vlasovets/atacama-soil-microbiome-tutorial): Full tutorial and example analyses

---

## License

MIT License. See [LICENSE](./LICENSE) for details.