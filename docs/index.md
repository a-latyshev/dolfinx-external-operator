# External operators for FEniCSx/DOLFINx

Authors:
* Andrey Latyshev (University of Luxembourg, Sorbonne Université,
  andrey.latyshev@uni.lu)
* Jérémy Bleyer (École des Ponts ParisTech, jeremy.bleyer@enpc.fr)
* Jack S. Hale (University of Luxembourg, jack.hale@uni.lu)
* Corrado Maurini (Sorbonne Université, corrado.maurini@sorbonne-universite.fr)

## About the project

This webpage aims to teach DOLFINx users to use external operators implemented
within the ... framework.

The concept of an external operator as a part of the Unified Form Language (UFL)
was originally introduced in {cite}`bouziani_escaping_2021`.

## Installation

This project extends the functionality of DOLFINx, so please ensure that you
installed the [FEniCSx environment](https://fenicsproject.org/download/) first.

Installation of the `dolfinx-external-operator` package:

```Shell
pip install .
```

## Acknowledgments & license

This research was funded in whole, or in part, by the Luxembourg National
Research Fund (FNR), grant reference PRIDE/21/16747448/MATHCODA. For the purpose
of open access, and in fulfilment of the obligations arising from the grant
agreement, the author has applied a Creative Commons Attribution 4.0
International (CC BY 4.0) license to any Author Accepted Manuscript version
arising from this submission.

This project is created using the open source Jupyter Book project and the book
of dolfinx-tutorial as a template.


```{bibliography}
:filter: docname in docnames
```