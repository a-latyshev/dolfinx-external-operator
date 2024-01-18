# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plasticity of von Mises
#
# ## Problem formulation
#
# ## Implementation
#
# ### Preamble

# %%
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import basix
from dolfinx import fem, common
from dolfinx.io import gmshio
import gmsh

import numba
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
# import fenicsx_support as fs
import solvers
# import classic_plasitcity

from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

# from external_operator_plasticity import build_mesh

# %%
R_e, R_i = 1.3, 1.   # external/internal radius

# elastic parameters
E = 70e3
nu = 0.3
lambda_ = E*nu/(1+nu)/(1-2*nu)
mu_ = E/2./(1+nu)

sig0 = 250.  # yield strength
Et = E/100.  # tangent modulus
H = E*Et/(E-Et)  # hardening modulus

TPV = np.finfo(PETSc.ScalarType).eps # très petite value 

q_lim = float(2/np.sqrt(3)*np.log(R_e/R_i)*sig0)

SQRT2 = np.sqrt(2.)

# %%
# Source: https://newfrac.github.io/fenicsx-fracture/notebooks/plasticity/plasticity.html
# mesh parameters
gdim = 2
lc = 0.3
verbosity = 0

# mesh using gmsh
mesh_comm = MPI.COMM_WORLD
model_rank = 0
gmsh.initialize()

facet_tags_labels = {"Lx": 1, "Ly": 2, "inner": 3, "outer": 4}

cell_tags_map = {"all": 20}
if mesh_comm.rank == model_rank:
    model = gmsh.model()
    model.add("Quart_cylinder")
    model.setCurrent("Quart_cylinder")
    # Create the points
    pix = model.occ.addPoint(R_i, 0.0, 0, lc)
    pex = model.occ.addPoint(R_e, 0, 0, lc)
    piy = model.occ.addPoint(0., R_i, 0, lc)
    pey = model.occ.addPoint(0., R_e, 0, lc)
    center = model.occ.addPoint(0., 0., 0, lc)
    # Create the lines
    lx = model.occ.addLine(pix, pex, tag = facet_tags_labels["Lx"])
    lout = model.occ.addCircleArc(pex, center, pey, tag = facet_tags_labels["outer"])
    ly = model.occ.addLine(pey, piy, tag = facet_tags_labels["Ly"])
    lin = model.occ.addCircleArc(piy, center, pix, tag = facet_tags_labels["inner"])
    # Create the surface
    cloop1 = model.occ.addCurveLoop([lx, lout, ly, lin])
    surface_1 = model.occ.addPlaneSurface([cloop1], tag = cell_tags_map["all"])
    model.occ.synchronize()      
    # Assign mesh and facet tags
    surface_entities = [entity[1] for entity in model.getEntities(2)]
    model.addPhysicalGroup(2, surface_entities, tag=cell_tags_map["all"])    
    model.setPhysicalName(2, 2, "Quart_cylinder surface")
    for (key, value) in facet_tags_labels.items():
            model.addPhysicalGroup(1, [value], tag=value) # 1 : it is the dimension of the object (here a curve)
            model.setPhysicalName(1, value, key)
    # Finalize mesh
    model.occ.synchronize()              
    gmsh.option.setNumber('General.Verbosity', verbosity)
    model.mesh.generate(gdim)

# import the mesh in fenicsx with gmshio
mesh, cell_tags, facet_tags = gmshio.model_to_mesh(gmsh.model, mesh_comm, 0., gdim=2)

mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
mesh.name = "Quart_cylinder"
cell_tags.name = f"{mesh.name}_cells"
facet_tags.name = f"{mesh.name}_facets"

# %%
deg_u = 2
deg_stress = 2
V = fem.VectorFunctionSpace(mesh, ("CG", deg_u))
We = ufl.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=4, quad_scheme='default')
W0e = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
W = fem.FunctionSpace(mesh, We)
W0 = fem.FunctionSpace(mesh, W0e)

# %%
