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


Before the release of SciPy 1.9.0, the need for these procedures was partially met in the scientific Python ecosystem by tutorials (e.g. blog posts, medium.com) and niche packages, but the functions in SciPy have several advantages:

- Prevalence: SciPy is one of the most downloaded scientific Python packages. If a Python user finds the need for these statistical methods, chances are that they already have SciPy, eliminating the need to find and install a new package.
- Speed: the functions take advantage of vectorized user code, avoiding slow Python loops.
- Easy-of-use: the function API reference and tutorials are thorough, and the interfaces share common features and complement other SciPy functions.
- Dependability: as with all SciPy code, these functions were rigorously peer-reviewed, and unit tests are extensive.

# Acknowledgements

The authors acknowledge Dr Sergei Kucherenko from Imperial College London and for helpful discussions.

The authors acknowledge Professors Fred Hickernell and Sergei Kucherenko for helpful discussions.
Dr Maximilian Balandat raised the initial issue which lead to these developments and provided the initial Cython implementation of the Sobol' method.
Lastly, the SciPy maintainer team provided support and help regarding the design and integration.

# References