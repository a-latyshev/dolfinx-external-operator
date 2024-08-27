# External operators within FEniCSx/DOLFINx

Authors:
* Andrey Latyshev (University of Luxembourg, Sorbonne Université,
  andrey.latyshev@uni.lu)
* Jérémy Bleyer (École des Ponts ParisTech, Université Gustave Eiffel, jeremy.bleyer@enpc.fr)
* Jack S. Hale (University of Luxembourg, jack.hale@uni.lu)
* Corrado Maurini (Sorbonne Université, corrado.maurini@sorbonne-universite.fr)

## About the project

This project introduces the `dolfinx-external-operator` framework that
implements an *external operator* concept within the DOLFINx library. It allows
to assemble forms containing symbolic objects that are not expressible via
Unified Form Language (UFL) and lets the user define the behaviour of these
objects and their derivatives using third-party packages. The data manipulation
between external software and DOLFINx is performed through `ndarray`-like
objects of the `Numpy` package.

Although the framework may be applied to a wide range of scientific problems,
during the development of our framework we were mainly focused on expressing
general constitutive models of solid mechanics problems in FEniCS. This webpage
contains several tutorials on plasticity problems.

In particular, the framework allows the DOLFINx library to use automatic
differentiation (AD). See this tutorial for further details, where the
constitutive model of the plasticity problem is solved via the `JAX` package, a
robust AD software.

The framework is based on the UFL extension proposed by
{cite:t}`bouzianiEscaping2021`, where the concept of the *external operator*
was originally introduced.

## Section table of contents

```{tableofcontents}
```

## Installation

This project extends the functionality of DOLFINx, so please ensure that you
install the [FEniCSx environment](https://fenicsproject.org/download/) first.

Installation of the `dolfinx-external-operator` package:

```Shell
pip install .
```

## Acknowledgments

This research was funded in whole, or in part, by the Luxembourg National
Research Fund (FNR), grant reference PRIDE/21/16747448/MATHCODA.

This project is created using the open-source
[Jupyter Book project](https://jupyterbook.org/en/stable/intro.html) and the
webpage of [dolfinx-tutorial](https://jsdokken.com/dolfinx-tutorial/index.html)
as a template.


```{bibliography}
:filter: docname in docnames
```