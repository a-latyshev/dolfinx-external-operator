from petsc4py import PETSc

import dolfinx.fem.petsc  # noqa: F401
import ufl
from dolfinx import fem, common
from typing import List, Union, Dict, Optional, Callable, Tuple

import pandas as pd
import numpy as np

class LinearProblem:
    def __init__(self, dR: ufl.Form, R: ufl.Form, u: fem.Function, bcs: list[fem.dirichletbc] | None = None):
        self.u = u
        self.bcs = bcs if bcs is not None else []

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

class SNESProblem:
    """Solves a nonlinear problem via PETSc.SNES.
    F(u) = 0
    J = dF/du
    b = assemble(F)
    A = assemble(J)
    Ax = b
    """
    def __init__(
        self,
        u: fem.Function,
        F: ufl.Form,
        J: ufl.Form,
        bcs=[],
        petsc_options={},
        prefix=None,
        system_update: Optional[Callable] = None,
    ):
        self.u = u
        V = self.u.function_space
        self.comm = V.mesh.comm

        self.F_form = fem.form(F)
        # J = ufl.derivative(F_form, self.u, ufl.TrialFunction(V))
        self.J_form = fem.form(J)
        self.b = fem.petsc.create_vector(self.F_form)
        self.A = fem.petsc.create_matrix(self.J_form)

        self.bcs = bcs

        # Give PETSc solver options a unique prefix
        if prefix is None:
            prefix = "snes_{}".format(str(id(self))[0:4])

        self.prefix = prefix
        self.petsc_options = petsc_options
        self.system_update = system_update

        self.solver = self.solver_setup()

        self.local_monitor = {"matrix_assembling": 0.0, "vector_assembling": 0.0, "constitutive_model_update": 0.0}
        self.performance_monitor = pd.DataFrame({
            # "loading_step": np.array([], dtype=np.int64),
            "Newton_iterations": np.array([], dtype=np.int64),
            "matrix_assembling": np.array([], dtype=np.float64),
            "vector_assembling": np.array([], dtype=np.float64),
            "nonlinear_solver": np.array([], dtype=np.float64),
            "constitutive_model_update": np.array([], dtype=np.float64),
        })
        self.timer = common.Timer("SNES")

    def set_petsc_options(self):
        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(self.prefix)

        for k, v in self.petsc_options.items():
            opts[k] = v

        opts.prefixPop()

    def solver_setup(self):
        # Create nonlinear solver
        snes = PETSc.SNES().create(self.comm)

        snes.setOptionsPrefix(self.prefix)
        self.set_petsc_options()
        snes.setFromOptions()

        snes.setFunction(self.F, self.b)
        snes.setJacobian(self.J, self.A)

        return snes

    def F(self, snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
        """Assemble the residual F into the vector b.

        Parameters
        ==========
        snes: the snes object
        x: Vector containing the latest solution.
        b: Vector to assemble the residual into.
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.scatter_forward()

        #TODO: SNES makes the iteration #0, where it calculates the b norm.
        #`system_update()` can be omitted in that case
        self.timer.start()
        self.system_update()
        self.timer.stop()
        self.local_monitor["constitutive_model_update"] += self.timer.elapsed()[0]

        self.timer.start()
        with b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(b, self.F_form)

        fem.petsc.apply_lifting(b, [self.J_form], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.bcs, x, -1.0)
        self.timer.stop()
        self.local_monitor["vector_assembling"] += self.timer.elapsed()[0]

    def J(self, snes, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """
        self.timer.start()
        A.zeroEntries()
        fem.petsc.assemble_matrix(A, self.J_form, self.bcs)
        A.assemble()
        self.timer.stop()
        self.local_monitor["matrix_assembling"] += self.timer.elapsed()[0]

    def solve(self,) -> Tuple[int, int]:
        # self.local_monitor["loading_step"] = loading_step
        self.local_monitor["vector_assembling"] = 0.0
        self.local_monitor["matrix_assembling"] = 0.0
        self.local_monitor["constitutive_model_update"] = 0.0
        timer = common.Timer("nonlinear_solver")
        self.timer.start()
        self.solver.solve(None, self.u.x.petsc_vec)
        timer.stop()
        self.local_monitor["nonlinear_solver"] = timer.elapsed()[0]
        self.local_monitor["Newton_iterations"] = self.solver.getIterationNumber()
        self.u.x.scatter_forward()
        self.performance_monitor.loc[len(self.performance_monitor.index)] = self.local_monitor
        return (self.solver.getIterationNumber(), self.solver.getConvergedReason())

    def __del__(self):
        self.solver.destroy()
        self.b.destroy()
        self.A.destroy()
