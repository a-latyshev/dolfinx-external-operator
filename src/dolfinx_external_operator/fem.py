"""Overloading of dolfinx Function spaces to add dual space functionality."""

# Copyright (C) 2009-2025 Chris N. Richardson, Garth N. Wells,
# Michal Habera and Jørgen S. Dokken
#
# This file is a modified version of python/dolfinx/fem/function.py which is part of
# the DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np

import basix
import dolfinx
import ufl

__all__ = ["DualSpace", "FunctionSpace", "functionspace"]


class FunctionSpace(dolfinx.fem.FunctionSpace):
    def dual(self):
        return DualSpace(self.ufl_domain(), self.ufl_element(), label=self.label, primal_space=self)

    def __hash__(self):
        return hash((self.ufl_domain(), self.ufl_element()))


class DualSpace(ufl.functionspace.DualSpace):
    def __init__(self, domain, element, label="", primal_space: dolfinx.fem.FunctionSpace | None = None):
        """Initialise."""
        super().__init__(domain, element, label)
        self._primal_space = primal_space

    def dual(self) -> FunctionSpace:
        return self._primal_space


def functionspace(
    mesh: dolfinx.mesh.Mesh,
    element: (
        ufl.finiteelement.AbstractFiniteElement
        | dolfinx.fem.ElementMetaData
        | tuple[str, int]
        | tuple[str, int, tuple]
        | tuple[str, int, tuple, bool]
    ),
) -> FunctionSpace:
    """Create a finite element function space.

    Args:
        mesh: Mesh that space is defined on.
        element: Finite element description.

    Returns:
        A function space.
    """
    # Create UFL element
    dtype = mesh.geometry.x.dtype
    try:
        e = dolfinx.fem.ElementMetaData(*element)  # type: ignore
        ufl_e = basix.ufl.element(
            e.family,
            mesh.basix_cell(),  # type: ignore
            e.degree,
            shape=e.shape,
            symmetry=e.symmetry,
            dtype=dtype,
        )
    except TypeError:
        ufl_e = element  # type: ignore

    # Check that element and mesh cell types match
    if ((domain := mesh.ufl_domain()) is None) or ufl_e.cell != domain.ufl_cell():
        raise ValueError("Non-matching UFL cell and mesh cell shapes.")
    # Create DOLFINx objects
    element = dolfinx.fem.finiteelement(mesh.topology.cell_type, ufl_e, dtype)  # type: ignore
    cpp_dofmap = dolfinx.cpp.fem.create_dofmap(mesh.comm, mesh.topology._cpp_object, element._cpp_object)  # type: ignore
    assert np.issubdtype(mesh.geometry.x.dtype, element.dtype), (  # type: ignore
        "Mesh and element dtype are not compatible."
    )

    # Initialize the cpp.FunctionSpace
    try:
        cppV = dolfinx.cpp.fem.FunctionSpace_float64(mesh._cpp_object, element._cpp_object, cpp_dofmap)  # type: ignore
    except TypeError:
        cppV = dolfinx.cpp.fem.FunctionSpace_float32(mesh._cpp_object, element._cpp_object, cpp_dofmap)  # type: ignore

    return FunctionSpace(mesh, ufl_e, cppV)
