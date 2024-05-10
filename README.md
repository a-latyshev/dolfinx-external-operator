![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)

# dolfinx-external-operator

`dolfinx-external-operator` is a framework implementing the [concept of external
operators](https://doi.org/10.48550/arXiv.2111.00945) within
[DOLFINx](https://github.com/FEniCS/dolfinx) and aims to define constitutive
models of solid mechanics via third-party software in FEniCSx.

Additionally, the framework shows how DOLFINx can be extended by enabling the general
automatic differentiation (AD) techniques through the
[JAX](https://jax.readthedocs.io/en/latest) library.

## Documentation

In the [documentation](https://a-latyshev.github.io/dolfinx-external-operator/), you can find the implementation of
* von Mises plasticity via [Numba](https://numba.pydata.org/),
* Mohr-Coulomb plasticity with apex smoothing via [JAX](https://jax.readthedocs.io/en/latest).

## Installation
From the source directory:

```Shell
pip install .
```

## Building Documentation

```Shell
pip install .[docs]
cd build/
jupyter-book build .
```

and follow the instructions printed.

## Development

To continuously build and view the documentation in a web browser

```Shell
pip install sphinx-autobuild
cd build/
jupyter-book config sphinx .
sphinx-autobuild . _build/html -b html
```

To check and fix formatting

```Shell
pip install `.[lint]`
ruff check .
ruff format .
```

## Citation 

If you use `dolfinx-external-operator` in your research, please cite the following publication:

```
@inproceedings{ORBi-bd33839e-f32f-4005-b706-a618d5bf1b0d,
	AUTHOR = {Latyshev, Andrey and Bleyer, Jérémy and Hale, Jack and Maurini, Corrado},
	TITLE = {A framework for expressing general constitutive models in FEniCSx},
	LANGUAGE = {English},
    BOOKTITLE = {CSMA 2024},
	YEAR = {February 2024},
	SIZE = {8},
	LOCATION = {Giens, France},
}
```