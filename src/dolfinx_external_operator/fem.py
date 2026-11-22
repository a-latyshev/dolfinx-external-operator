"""Overloading of dolfinx Function spaces to add dual space functionality."""

# Copyright (C) 2009-2025 Chris N. Richardson, Garth N. Wells,
# Michal Habera and Jørgen S. Dokken
#
# This file is a modified version of python/dolfinx/fem/function.py which is part of
# the DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import ufl

__all__ = ["DualSpace", "FunctionSpace", "functionspace"]


class FunctionSpace(dolfinx.fem.FunctionSpace):
    def dual(self):
        return DualSpace(self.ufl_domain(), self.ufl_element(), label=self.label(), primal_space=self)

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
    space = dolfinx.fem.functionspace(mesh, element)
    return FunctionSpace(space.mesh, space.ufl_element(), space._cpp_object)
