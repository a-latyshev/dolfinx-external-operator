from .external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)
from .fem import DualSpace, FunctionSpace, functionspace
from .petsc import petsc

__all__ = [
    "DualSpace",
    "FEMExternalOperator",
    "FunctionSpace",
    "evaluate_external_operators",
    "evaluate_operands",
    "functionspace",
    "petsc",
    "replace_external_operators",
]
