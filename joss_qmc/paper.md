---
title: 'Quasi-Monte Carlo Methods in Python'
tags:
  - Python
  - SciPy
  - statistics
  - Quasi-Monte Carlo methods
authors:
  - name: Pamphile T. Roy
    affiliation: 1
    corresponding: true
    orcid: 0000-0001-9816-1416
  - name: Art B. Owen
    affiliation: 1
    orcid: 0000-0001-5860-3945
  - name: Matt Haberland
    affiliation: 3
    orcid: 0000-0003-4806-3601
affiliations:
    - name: Quansight
      index: 1
    - name: Stanford University
      index: 2
    - name: California Polytechnic State University, San Luis Obispo, USA
      index: 3
date: 20 March 2022
bibliography: paper.bib

---

# Summary


# Statement of need

NumPy random number generators (`numpy.random`) has become the standard way of
sampling random numbers in the scientific Python ecosystem.
These methods are fast and reliable, and the results are repeatable when a
random seed is provided. However, sampling in high dimensions produces a lot of
gaps and clusters of points. In integration problems, classical methods have a 
low convergence rate meaning that a large sample size is required.

By construction, Quasi-Monte Carlo (QMC) methods provide efficient, determinist
(or not) and quality generators that can advantageously replace traditional
methods. This can be decisive when the sampling size is limited or strong
reproducibility guarantee is required. Also QMC methods are known to have
better convergence rate than traditional Monte Carlo sampling (used by NumPy).

QMC methods where added to SciPy [@virtanen2020scipy] after an extensive review
and discussion period [@scipy2021qmc] that lead to a great collaboration
between SciPy's maintainers and renown researchers in the field.

Before the release of SciPy 1.7.0, the need for these functions was partially
met in the scientific Python ecosystem by tutorials (e.g. blog posts)
and niche packages, but the functions in SciPy have several advantages:

- Prevalence: SciPy is one of the most downloaded scientific Python packages.
  If a Python user finds the need for these statistical methods, chances are
  that they already have SciPy, eliminating the need to find and install a
  new package.
- Speed: low level functions are written and optimized in Cython.
- Easy-of-use: the function API reference and tutorials are thorough,
  and the interfaces share common features and complement other SciPy functions.
- Dependability: as with all SciPy code, these functions were rigorously
  peer-reviewed, and unit tests are extensive.

Since the first release of all these new features, we have seen other libraries
add support for and rely on SciPy's implementations,
e.g. [@optuna2022qmc; @salib2022qmc].

# Acknowledgements

The authors acknowledge Dr Sergei Kucherenko from Imperial College London and
for helpful discussions. The authors acknowledge Professors Fred Hickernell and
Sergei Kucherenko for helpful discussions. Dr Maximilian Balandat raised the
initial issue which lead to these developments and provided the initial Cython
implementation of the Sobol' method. Lastly, the SciPy maintainer team provided
support and help regarding the design and integration.

# References