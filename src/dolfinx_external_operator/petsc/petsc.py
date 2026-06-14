from collections.abc import Callable, Sequence

from petsc4py import PETSc

from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.function import Function
from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc


def assemble_residual_with_callback(
    u: Function,
    F: Form,
    J: Form,
    bcs: Sequence[DirichletBC],
    external_callback: Callable,
    args_external_callback: Sequence,
    snes: PETSc.SNES,
    x: PETSc.Vec,
    b: PETSc.Vec,
) -> None:
    """Assemble the residual at ``x`` into the vector ``b`` with a callback to
    external functions.

    Prior to assembling the residual and after updating the solution ``u``, the
    function ``external_callback`` with input arguments ``args_external_callback``
    is called.

    A function conforming to the interface expected by ``SNES.setFunction`` can
    be created by fixing the first 5 arguments, e.g. (cf.
    ``dolfinx.fem.petsc.assemble_residual``):

    Example::

        snes = PETSc.SNES().create(mesh.comm)
        assemble_residual = functools.partial(
            dolfinx.fem.petsc.assemble_residual, u, F, J, bcs,
            external_callback, args_external_callback)
        snes.setFunction(assemble_residual, b)

    Args:
        u: Function tied to the solution vector within the residual and
           Jacobian.
        F: Form of the residual.
        J: Form of the Jacobian.
        bcs: List of Dirichlet boundary conditions to lift the residual.
        external_callback: A callback function to call prior to assembling the
                           residual.
        args_external_callback: Arguments to pass to the external callback
                                function.
        snes: The solver instance.
        x: The vector containing the point to evaluate the residual at.
        b: Vector to assemble the residual into.
    """
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    x.copy(u.x.petsc_vec)
    u.x.scatter_forward()

    # Call external functions, e.g. evaluation of external operators
    external_callback(*args_external_callback)

    with b.localForm() as b_local:
        b_local.set(0.0)
    assemble_vector(b, F)

    apply_lifting(b, [J], [bcs], [x], -1.0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs, x, -1.0)
