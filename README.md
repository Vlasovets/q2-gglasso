# q2-gglasso


[![PyPI license](https://img.shields.io/pypi/l/gglasso.svg)](https://pypi.python.org/pypi/gglasso/)

[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)

[![Documentation Status](https://readthedocs.org/projects/gglasso/badge/?version=latest)](http://gglasso.readthedocs.io/?badge=latest)


This is a QIIME 2 plugin which contains algorithms for solving General Graphical Lasso (GGLasso) problems, including single, multiple, as well as latent 

Graphical Lasso problems. <br>


[Docs](https://gglasso.readthedocs.io/en/latest/) | [Examples](https://gglasso.readthedocs.io/en/latest/auto_examples/index.html)


For details on QIIME 2, see https://qiime2.org.


# Installation


# Tutorial
Welcome to this tutorial on using QIIME 2 for analyzing soil samples from the Atacama Desert in 
northern Chile. This tutorial assumes that you have already installed QIIME 2, but if you 
haven't, you can follow the instructions from the [docs](https://docs.qiime2.org/2022.11/install/).

The Atacama Desert is known to be one of the most arid locations on Earth, with some areas receiving 
less than a millimeter of rain per decade. Despite such extreme conditions, the soil in the Atacama Desert 
is known to harbor a variety of microbial life. In this tutorial, we will explore how to use 
graphical models for analyzing microbial compositions in soil samples from the Atacama Desert.

Specifically, we will demonstrate the application of Single graphical lasso (SGL), adaptive 
SGL, and SGL + low-rank, to illustrate how covariates are related to microbial compositions.

Let's get started!

## Compositional data
In the following tutorial we will work with 130 ASVs written into count table of the following 
format:

|       | ASV_1     | ASV_2     | ASV_3     | ASV_4     | ASV_5     |
|-------|-----------|-----------|-----------|-----------|-----------|
| ASV_1 | 1.000000  | 0.235220  | -0.300181 | 0.125571  | 0.179510  | 
| ASV_2 | 0.235220  | 1.000000  | 0.115052  | -0.113347 | 0.271909  | 
| ASV_3 | -0.300181 | 0.115052  | 1.000000  | -0.177694 | -0.238362 |
| ASV_4 | 0.125571  | -0.113347 | -0.177694 | 1.000000  | 0.146477  |
| ASV_5 | 0.179510  | 0.271909  | -0.238362 | 0.146477  | 1.000000  | 

Please note that preprocessing steps, such as the [center log-ratio transformation](https://en.wikipedia.org/wiki/Compositional_data#:~:text=in%20the%20simplex.-,Center%20logratio%20transform,-%5Bedit%5D) 
of the count table and [scaling](https://en.wikipedia.org/wiki/Feature_scaling) metadata, have 
been omitted from this tutorial, but you can find these steps in the documentation linked [here](https://github.com/Vlasovets/atacama-soil-microbiome-tutorial/blob/main/python/tutorial.ipynb).

![covariates](./example/atacama/plots/asv_correlation.png)
Figure 1. Correlation between ASVs in Atacama soil microbiome.

## Metadata
This section presents a description and basic statistical analysis of the covariates 
under investigation. For more comprehensive information about 
the research, readers are referred to the original [paper](https://www.frontiersin.org/articles/10.3389/fmicb.2021.794743/full).


| feature                            | mean  | std     | min  | max   | description (units)                      |  
|------------------------------------|-------|---------|------|-------|------------------------------------------|
| elevation                          | 2825  | 1014.23 | 895  | 4697  | meters above sea level (m.a.s.l.)        |
| extract-concen                     | 2.92  | 5.96    | 0.01 | 33.49 | (µg/ml)                                  |
| amplicon-concentration             | 9.54  | 6.81    | 0.12 | 19.2  | µg/ml                                    |
| depth                              | 2     | 0.46    | 1    | 3     | range 0–60 / 60–220 / 220–340 (cm)       |
| ph                                 | 7.05  | 2.53    | 0    | 9.36  | potential of hydrogen (log)              |
| toc                                | 693.8 | 1958.49 | 0    | 16449 | total organic carbon (μg/g)              |
| ec                                 | 0.72  | 1.26    | 0    | 6.08  | electric conductivity (S/m in SI)        |
| average-soil-relative-humidity     | 63.27 | 33.54   | 0    | 100   | average soil humidity (%)                |
| relative-humidity-soil-high        | 78.51 | 32.09   | 0    | 100   | %                                        |
| relative-humidity-soil-low         | 43.62 | 32.58 | 0     | 100   | %                                        |
| percent-relative-humidity-soil-100 | 37.86 | 39.45 | 0     | 100   | %                                        |
| average-soil-temperature           | 15.72 | 5.8   | 0     | 23.61 | average soil temperature (t°)            |
| temperature-soil-high              | 23.61 | 6.82  | 0     | 35.21 | t°                                       |
| temperature-soil-low               | 7.24  | 5.96  | -2.57 | 18.33 | t°                                       |
| percentcover                       | 1.82  | 3.05  | 0     | 8.8   | vegetation coverage per square meter (%) |

Figure 2 illustrates the correlation between the covariates, it is clear that some of them are 
highly correlated and thus can be disregarded.
For instance, in the subsequent analysis of humidity and temperature, their average values shall suffice for our purposes.
![covariates](./example/atacama/plots/covariates_correlation.png)
Figure 2. Correlation between Atacama covariates.

## Analysis
### SGL
![covariates](./example/atacama/plots/SGL.png)
Figure 3. Single graphical lasso problem solution.

![covariates](./example/atacama/plots/SGL_network.png)
Figure 3. Single graphical lasso problem solution.


### Adaptive SGL

![covariates](./example/atacama/plots/lambda_mask.png)
Figure 4. Single graphical lasso problem solution.

![covariates](./example/atacama/plots/adapt_SGL.png)
Figure 4. Single graphical lasso problem solution.

![covariates](./example/atacama/plots/adapt_network.png)
Figure 3. Single graphical lasso problem solution.

### SGL + low-rank
![covariates](./example/atacama/plots/low_rank.png)
Figure 5. Single graphical lasso + low-rank problem solution.

![covariates](./example/atacama/plots/low_network.png)
Figure 3. Single graphical lasso problem solution.