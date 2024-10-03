from typing import List

from petsc4py import PETSc

import dolfinx.fem.petsc  # noqa: F401
import ufl
from dolfinx import fem


class LinearProblem:
    def __init__(self, dR: ufl.Form, R: ufl.Form, u: fem.Function, bcs: List[fem.dirichletbc] = []):
        self.u = u
        self.bcs = bcs

        V = u.function_space
        domain = V.mesh

        self.R = R
        self.dR = dR
        self.b_form = fem.form(R)
        self.A_form = fem.form(dR)
        self.b = fem.petsc.create_vector(self.b_form)
        self.A = fem.petsc.create_matrix(self.A_form)

        self.comm = domain.comm

        self.solver = self.solver_setup()

    def solver_setup(self) -> PETSc.KSP:
        """Sets the solver parameters."""
        solver = PETSc.KSP().create(self.comm)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.setOperators(self.A)
        return solver

    def assemble_vector(self) -> None:
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(self.b, self.b_form)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(self.b, self.bcs)

    def assemble_matrix(self) -> None:
        self.A.zeroEntries()
        fem.petsc.assemble_matrix(self.A, self.A_form, bcs=self.bcs)
        self.A.assemble()

    def assemble(self) -> None:
        self.assemble_matrix()
        self.assemble_vector()

    def solve(
        self,
        du: fem.function.Function,
    ) -> None:
        """Solves the linear system and saves the solution into the vector `du`

        Args:
            du: A global vector to be used as a container for the solution of the linear system
        """
        self.solver.solve(self.b, du.x.petsc_vec)

    def __del__(self):
        self.solver.destroy()
        self.A.destroy()
        self.b.destroy()
