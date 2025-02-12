from typing import Callable, Optional

from petsc4py import PETSc

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem


class LinearProblem:
    def __init__(
        self, dR: ufl.Form, R: ufl.Form, u: fem.function.Function, bcs: list[fem.bcs.DirichletBC] | None = None
    ):
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
        pc = solver.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        solver.setOperators(self.A)
        return solver

    def assemble_vector(self) -> None:
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(self.b, self.b_form)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.apply_lifting(self.b, [self.A_form], bcs=[self.bcs], x0=[self.u.x.petsc_vec], alpha=-1.0)
        fem.petsc.set_bc(self.b, self.bcs, self.u.x.petsc_vec, -1.0)
        self.b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

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


class NewtonProblem(NonlinearProblem):
    """Problem for the DOLFINx NewtonSolver with an external callback."""

    def __init__(
        self,
        F: ufl.form.Form,
        u: fem.function.Function,
        bcs: list[fem.bcs.DirichletBC] = [],
        J: ufl.form.Form = None,
        form_compiler_options: Optional[dict] = None,
        jit_options: Optional[dict] = None,
        external_callback: Optional[Callable] = None,
    ):
        super().__init__(F, u, bcs, J, form_compiler_options, jit_options)

        self.external_callback = external_callback

    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values, but here
        we also use it to evaluate the external operators.

        Args:
           x: The vector containing the latest solution
        """
        # The following line is from the standard NonlinearProblem class
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # The external operators are evaluated here
        self.external_callback()


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
        u: fem.function.Function,
        F: ufl.form.Form,
        J: ufl.form.Form,
        bcs: list[fem.bcs.DirichletBC] = [],
        petsc_options: Optional[dict] = {},
        prefix: Optional[str] = None,
        external_callback: Optional[Callable] = None,
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
            prefix = f"snes_{str(id(self))[0:4]}"

        self.prefix = prefix
        self.petsc_options = petsc_options
        self.external_callback = external_callback

        self.solver = self.solver_setup()

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
        # snes.setUpdate(self.update)

        return snes

    def update(self, snes: PETSc.SNES, iter: int) -> None:
        """Call external function at each iteration."""
        self.external_callback()

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

        self.external_callback()

        with b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(b, self.F_form)

        fem.petsc.apply_lifting(b, [self.J_form], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.bcs, x, -1.0)

    def J(self, snes, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """
        A.zeroEntries()
        fem.petsc.assemble_matrix(A, self.J_form, self.bcs)
        A.assemble()

    def solve(
        self,
    ) -> tuple[int, int]:
        self.solver.solve(None, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        return (self.solver.getIterationNumber(), self.solver.getConvergedReason())

    def __del__(self):
        self.solver.destroy()
        self.b.destroy()
        self.A.destroy()
