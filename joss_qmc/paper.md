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

NumPy random number generators (`numpy.random`) have become the de-facto
standard for sampling random numbers in the scientific Python ecosystem.
These methods are fast and reliable, and the results are repeatable when a
seed is provided. As a foundational tool, NumPy only provides classical
Monte Carlo (MC) methods. Sampling in high dimensions with MC produces a lot of
gaps and clusters of points. When these random numbers are used in algorithms
to solve deterministic problems, the resulting MC methods have a low
convergence rate. In practice, this can mean that substantial computational
resources are required to provide sufficient accuracy.

In Quasi-Monte Carlo (QMC) methods [@owen2019], the random numbers of Monte
Carlo methods are replaced with a deterministic sequence of numbers that
possesses many of the characteristics of a random sequence
(e.g. reduction of variance with the sample size), but without these gaps
and clusters. QMC determinism is implementation, language and platform,
independent. The sequence is mathematically defined. 

In many cases, a QMC sequence can be used as a drop-in
replacement for a random number sequence, yet they are proven to provide faster
convergence rates. When true stochasticity is required (e.g. statistical
inference), QMC sequences can be "scrambled" using random numbers, and several
smaller scrambled QMC sequences can often replace one large random sequence.

QMC methods were added to SciPy [@virtanen2020scipy] after an extensive review
and discussion period [@scipy2021qmc] that lead to a very fruitful collaboration
between SciPy's maintainers and renowned researchers in the field.
Our implementation work inspired additional work on highlighting the importance
of including the first point in the Sobol' sequence [owen2020].

The following set of QMC features are currently available in SciPy:

- Sobol' and Halton sequences (scrambled and unscrambled),
- Poisson disk sampling,
- Quasi-random multinomial and multivariate normal sampling,
- Discrepancy measures ($C^2$, wrap around, star-$L_2$, mixed),
- Latin Hypercube Sampling (centred, strength 1 or 2),
- Optimize a sample by minimizing $C^2$ discrepancy or performing Lloyd-Max
  iterations,
- Scaling utilities,
- Fast numerical inverse methods to sample arbitrary distributions with QMC.
  Additional methods have been later added. They wrap the UNU.RAN library
  [unuran2022].

Before the release of SciPy 1.7.0, the need for these functions was partially
met in the scientific Python ecosystem by tutorials (e.g. blog posts)
and niche packages, but the functions in SciPy have several advantages:

- Popularity: with an estimated 5 million download per month, SciPy is one of
  the most downloaded scientific Python packages. New features immediately
  reach a wide range of users from all fields.
- Performance: The low level functions are written in compiled languages such
  as Cython and optimized for speed and efficiency.
- Consistency: The APIs comply with the high standards of SciPy, function API
  reference and tutorials are thorough, and the interfaces share common
  features complementing other SciPy functions.
- Quality: As with all SciPy code, these functions were rigorously
  peer-reviewed for code quality and are extensively unit-tested. In addition,
  the implementations were produced in collaboration with the foremost experts
  in the QMC field.

Since the first release of all these new features, we have seen other libraries
add support for and rely on SciPy's implementations,
e.g. Optuna [@optuna2022qmc] and SALib [@salib2022qmc].

# Acknowledgements

The authors thank professors Sergei Kucherenko (Imperial College London) and
Fred Hickernell (Illinois Institute of Technology) for helpful discussions.
The SciPy maintainer team provided support and help regarding the design and
integration, notably Ralf Gommers (Quansight) and
Tyler J. Reddy (Los Alamos National Laboratory).

# References