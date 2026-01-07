![License: LGPLv3](https://img.shields.io/badge/License-LGPL%20v3.0-lightgrey.svg)

# dolfinx-external-operator

`dolfinx-external-operator` is a implementation of the [external
operator](https://doi.org/10.48550/arXiv.2111.00945) concept in
[DOLFINx](https://github.com/FEniCS/dolfinx).

It allows for the expression of operators/functions in FEniCS that cannot be
easily written in the [Unified Form Language](https://github.com/fenics/ufl).

Potential application areas include complex constitutive models in solid and
fluid mechanics, neural network constitutive models, multiscale modelling and
inverse problems. 

Implementations of external operators can be written in any library that
supports the [array interface
protocol](https://numpy.org/doc/stable/reference/arrays.interface.html) e.g. 
[NumPy](https://numpy.org/), [JAX](https://github.com/google/jax) and
[Numba](http://numba.pydata.org).

When using a library that supports program level automatic differentiation
(AD), such as JAX, it is possible to automatically derive derivatives for use
in local first and second-order solvers. Just-in-time compilation, batching and
accelerators (GPUs, TPUs) are also supported.

## Installation

`dolfinx-external-operator` is a pure Python package that depends on the
DOLFINx Python interface and UFL. Version numbers match with compatible
releases of DOLFINx.

The latest release version can be installed with:

```Shell
pip install dolfinx-external-operator
```

The latest development version can be installed for development with:

```Shell
git clone https://github.com/a-latyshev/dolfinx-external-operator.git
cd dolfinx-external-operator
pip install -e .
```

## Documentation

The [documentation](https://a-latyshev.github.io/dolfinx-external-operator/)
contains various examples focusing on complex constitutive behaviour in solid
mechanics, including:

* von Mises plasticity using [Numba](https://numba.pydata.org/),
* Mohr-Coulomb plasticity using [JAX](https://jax.readthedocs.io/en/latest).

Introduction to External Operators — FEniCS 2024 Conference:

<iframe width="700" height="400" src="https://www.youtube.com/embed/y8goeapqfsw?si=BQwDcdwqS3EF5rwW" frameborder="0" allowfullscreen></iframe>

## Citations 

If you use `dolfinx-external-operator` in your research we ask that you cite
the following references:

```bibtex
@article{jtcam:14449,
    title      = {Expressing general constitutive models in FEniCSx using external operators and algorithmic automatic differentiation},
    author     = {Andrey Latyshev and Jérémy Bleyer and Corrado Maurini and Jack Hale},
    doi        = {10.46298/jtcam.14449},
    journal    = {Journal of Theoretical, Computational and Applied Mechanics},
    issn       = {2726-6141},
    year       = {2025},
    month      = {Sep},
    keywords   = {automated finite element solvers, Numba, JAX, FEniCSx, external operators, algorithmic automatic differentiation, constitutive models}
}

@inproceedings{latyshev_2024_external_paper,
  author = {Latyshev, Andrey and Bleyer, Jérémy and Hale, Jack and Maurini, Corrado},
  title = {A framework for expressing general constitutive models in FEniCSx},
  booktitle = {16ème Colloque National en Calcul de Structures},
  year = {2024},
  month = {May},
  publisher = {CNRS, CSMA, ENS Paris-Saclay, CentraleSupélec},
  address = {Giens, France},
  url = {https://hal.science/hal-04610881}
}

@software{latyshev_2024_external_code,
  title = {a-latyshev/dolfinx-external-operator},
  author = {Latyshev, Andrey and Hale, Jack},
  date = {2025},
  doi = {10.5281/zenodo.10907417},
  organization = {Zenodo}
}
```

## Contributors

* Andrey Latyshev (University of Luxembourg, Sorbonne Université,
  andrey.latyshev@uni.lu)
* Jérémy Bleyer (École des Ponts ParisTech, Université Gustave Eiffel, jeremy.bleyer@enpc.fr)
* Jack S. Hale (University of Luxembourg, jack.hale@uni.lu)
* Corrado Maurini (Sorbonne Université, corrado.maurini@sorbonne-universite.fr)
* Jørgen S. Dokken (Simula Research Laboratory, dokken@simula.no)

If you wish to be added as a contributor after an accepted PR please ask via
email.

## License

dolfinx-external-operator is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

dolfinx-external-operator is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
for more details.

You should have received a copy of the GNU Lesser General Public License along
with dolfinx-external-operator. If not, see https://www.gnu.org/licenses/.

## Acknowledgments

This research was funded in whole, or in part, by the Luxembourg National
Research Fund (FNR), grant reference PRIDE/21/16747448/MATHCODA.

## Developer notes

### Docker container

```Shell
docker run -ti -v $(pwd):/shared --entrypoint /bin/bash -w /shared docker.io/dolfinx/lab:nightly
pip install .
```

### Building Documentation

```Shell
pip install '.[doc]'
cd doc/
jupyter-book build .
```

and follow the instructions printed.

To continuously build and view the documentation in a web browser

```Shell
pip install sphinx-autobuild
cd build/
jupyter-book config sphinx .
sphinx-autobuild . _build/html -b html
```

### Linting

To lint and format

```Shell
pip install '.[lint]'
ruff check .
ruff format .
```

### Running tests

```Shell
pip install '.[test]'
py.test -v test/
```

### Releases

```Shell
git pull
git checkout release
git merge --no-commit origin/main
git checkout --theirs . # files deleted on `main` must be manually git `rm`ed
vim pyproject.toml # Update version numbers
git diff origin/main # Check for mistakes
git tag v0.9.0 # for example
git push --tags origin
```

Then make a release using GitHub Releases.
