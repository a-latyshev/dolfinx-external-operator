# dolfinx-external-operator

Extension of external operator concept of UFL

## Documentation

[Main webpage](https://a-latyshev.github.io/dolfinx-external-operator/intro.html)

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

To continuosly build and view the documentation in a web browser

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
