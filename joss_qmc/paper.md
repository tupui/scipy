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
    affiliation: 2
    orcid: 0000-0001-5860-3945
  - name: Maximilian Balandat
    affiliation: 3
    orcid: 0000-0002-8214-8935
  - name: Matt Haberland
    affiliation: 4
    orcid: 0000-0003-4806-3601
affiliations:
    - name: Quansight
      index: 1
    - name: Stanford University
      index: 2
    - name: Meta
      index: 3
    - name: California Polytechnic State University, San Luis Obispo, USA
      index: 4
date: 30 August 2022
bibliography: paper.bib

---

# Summary


# Statement of need

NumPy random number generators (`numpy.random`) have become the de-facto standard
for sampling random numbers in the scientific Python ecosystem.
These methods are fast and reliable, and the results are repeatable when a
random seed is provided. However, sampling in high dimensions produces a lot of
gaps and clusters of points. In integration problems, these classical "Monte Carlo"
methods have a low convergence rate meaning that a large sample size is required
to achieve good accuracy. 

By construction, Quasi-Monte Carlo (QMC) methods provide efficient, deterministic
(or not) high quality sample generators that can advantageously replace traditional
methods. This can be decisive when the sampling size is limited or strong
reproducibility guarantee is required. QMC methods are known to have better
convergence rates than traditional Monte Carlo sampling (as implmented by NumPy).

QMC methods were added to SciPy [@virtanen2020scipy] after an extensive review
and discussion period [@scipy2021qmc] that lead to a very fruitful collaboration
between SciPy's maintainers and renowned researchers in the field.

The following set of features QMC are currently available in SciPy:

- Sobol' and Halton sequences (scrambled and unscrambled),
- Poisson disk sampling,
- Quasi-random multinomial and multivariate normal sampling,
- Discrepancy measures ($C^2$, wrap around, star-$L_2$, mixed),
- Latin Hypercube Sampling (centred, optimized on $C^2$, orthogonal),
- Optimize a sample using $C^2$ or Lloyd-Max iterations,
- Scaling utilities,
- Fast numerical inverse methods to sample arbitrary distributions with QMC.

Before the release of SciPy 1.7.0, the need for these functions was partially
met in the scientific Python ecosystem by tutorials (e.g. blog posts)
and niche packages, but the functions in SciPy have several advantages:

- Popularity: with an estimated 5 million download per month, SciPy is one of the most downloaded scientific Python packages. New features immediatelly reach a wide range of users from all fields.
- Performance: The low level functions are written in compiled languages such as Cython and optimized for speed and efficiency.
- Consistency: The APIs comply with the high standards of SciPy, function API reference and tutorials are thorough, and the interfaces share common features complementing other SciPy functions.
- Quality: As with all SciPy code, these functions were rigorously peer-reviewed and are extensively unit-tested. In addition, theimplementation has been extensively tested in collaboration with the foremost experts in the field.

Since the first release of all these new features, we have seen other libraries
add support for and rely on SciPy's implementations,
e.g. [@optuna2022qmc; @salib2022qmc].

# Acknowledgements

The authors thank professors Sergei Kucherenko (Imperial College London) and
Fred Hickernell (Illinois Institute of Technology) for helpful discussions.
The SciPy maintainer team provided support and help regarding the design and integration.

# References