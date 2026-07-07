---
name: dolfinx-external-operator
description: >-
  Rules, guidelines, and API documentation for expressing non-UFL operators
  and functions in DOLFINx/FEniCSx using dolfinx-external-operator and
  algorithmic automatic differentiation (JAX, PyTorch, Numba, NumPy).
---

# AI Context & Agent Skill: dolfinx-external-operator

This document provides system instructions, design patterns, API structures, and troubleshooting tips for the `dolfinx-external-operator` library. AI models and coding assistants should read this file to generate correct, idiomatic code for this library.

---

## 1. Library Overview

`dolfinx-external-operator` implements the external operator concept in DOLFINx (FEniCSx). It allows users to express operators, equations, and constitutive models (e.g., neural network models, complex plasticity, or multiscale models) that cannot be easily written in the Unified Form Language (UFL). 

External functions can be implemented using any library supporting the Python array interface protocol (e.g., NumPy, Numba, JAX, PyTorch). When using JAX or PyTorch, program-level automatic differentiation (AD) is supported.

---

## 2. Core API Reference

The library exposes four primary user-facing functions/classes:

```python
from dolfinx_external_operator import (
    FEMExternalOperator,
    replace_external_operators,
    evaluate_operands,
    evaluate_external_operators,
)
```

### `FEMExternalOperator`
Inherits from `ufl.ExternalOperator`. It represents an external operator symbolically within a UFL form.

* **Signature**:
  ```python
  FEMExternalOperator(
      *operands,
      function_space: dolfinx.fem.FunctionSpace,
      external_function: Callable[[tuple[int, ...]], Callable] | None = None,
      derivatives: tuple[int, ...] | None = None,
      name: str | None = None,
      coefficient: dolfinx.fem.Function | None = None,
      argument_slots: tuple = (),
      dtype: numpy.typing.DTypeLike | None = None
  )
  ```
* **Key Arguments**:
  * `*operands`: The symbolic arguments the external operator acts on (e.g., `u`, `grad(u)`, `div(u)`).
  * `function_space`: The `dolfinx.fem.FunctionSpace` where the operator's outputs will reside. Typically a `Quadrature` or `DG` (Discontinuous Galerkin) space.
  * `external_function`: A double-callable Python function that maps derivative tuples to implementation functions (see section 4).
  * `derivatives`: A tuple of integers representing the order of derivatives with respect to each operand (defaults to all zeros).
  * `coefficient`: Pre-allocated `dolfinx.fem.Function` where outputs are stored. If `None`, one is created automatically.

---

## 3. The Execution & Evaluation Pipeline

Unlike built-in UFL operators, forms containing `FEMExternalOperator` cannot be compiled directly. They must go through a replacement and evaluation pipeline prior to compilation or assembly. 

### The Standard Compilation & Assembly Workflow

Always follow these exact steps when compiling or assembling forms:

```python
import ufl
from dolfinx import fem
from dolfinx_external_operator import (
    replace_external_operators,
    evaluate_operands,
    evaluate_external_operators,
)

# 1. Define the UFL form using the FEMExternalOperator
F = N * v * ufl.dx

# 2. Extract derivatives (if any) and replace external operators with standard coefficients
F_replaced, F_external_operators = replace_external_operators(F)

# 3. Evaluate the symbolic operands at the operator's interpolation points
evaluated_operands = evaluate_operands(F_external_operators)

# 4. Run the external function callback and update the coefficient values
_ = evaluate_external_operators(F_external_operators, evaluated_operands)

# 5. Compile the replaced form using DOLFINx
F_compiled = fem.form(F_replaced)

# 6. Assemble vector/matrix or solve as usual
b = fem.assemble_vector(F_compiled)
```

> [!IMPORTANT]
> Step 3 and 4 (`evaluate_operands` and `evaluate_external_operators`) must be executed **every time** the values of the operands (e.g., displacement or temperature fields) change, particularly during Newton solver iterations.

---

## 4. Writing External Functions (`external_function`)

The `external_function` parameter expects a **double-callable factory**:
1. It receives a `derivatives` tuple specifying the derivative multiindex (e.g. `(0, 0)` for no derivatives, `(1, 0)` for derivative w.r.t the first operand).
2. It returns the actual numerical implementation function.

### Array Interface & Shapes
All inputs passed to the numerical implementation function are structured cell-by-cell at the interpolation points of the operator's function space.
For `N` operands, the implementation function receives `N` arrays of shapes:
$$\text{Shape} = (\text{num\_cells}, \text{num\_interpolation\_points\_per\_cell}, *\text{operand\_value\_shape})$$

The implementation must return a flat NumPy array of size `(num_cells * num_points * output_value_size)`.

### Example: Nonlinear Heat Conductivity (NumPy)

```python
import numpy as np

def k_model(T):
    # T shape: (num_cells, num_points)
    # Output shape: (num_cells, num_points)
    return 1.0 / (1.0 + T)

def dk_dT_model(T):
    # Output shape: (num_cells, num_points)
    return -1.0 / (1.0 + T)**2

def my_external_function(derivatives):
    if derivatives == (0,):
        # Return implementation for k(T)
        def impl(T_eval):
            return k_model(T_eval).flatten()
        return impl
    elif derivatives == (1,):
        # Return implementation for dk/dT
        def impl(T_eval):
            return dk_dT_model(T_eval).flatten()
        return impl
    else:
        raise NotImplementedError
```

---

## 5. Algorithmic Automatic Differentiation (JAX & PyTorch)

When using libraries supporting Automatic Differentiation (JAX or PyTorch), you can write a single forward implementation and let the library compute the derivatives automatically.

### Example: Hyperelastic Model (JAX)

```python
import jax
import jax.numpy as jnp
import numpy as np

# 1. Define energy function for a single point
def energy_fn(F_tensor):
    # F_tensor is a 2x2 deformation gradient tensor
    C = F_tensor.T @ F_tensor
    I1 = jnp.trace(C)
    return 0.5 * (I1 - 2.0)

# 2. Define analytical derivatives using JAX
stress_fn = jax.grad(energy_fn)
tangent_fn = jax.jacobian(stress_fn)

# 3. Create vectorized versions for arrays of shapes (num_cells, num_points, 2, 2)
# vmap across cell axis (0) and point axis (0)
vectorized_stress = jax.vmap(jax.vmap(stress_fn))
vectorized_tangent = jax.vmap(jax.vmap(tangent_fn))

def hyperelastic_operator(derivatives):
    if derivatives == (0,):
        def impl(F_eval):
            # F_eval shape: (num_cells, num_points, 2, 2)
            out = vectorized_stress(F_eval)
            return np.array(out).flatten()
        return impl
    elif derivatives == (1,):
        def impl(F_eval):
            out = vectorized_tangent(F_eval)
            return np.array(out).flatten()
        return impl
    else:
        raise NotImplementedError
```

---

## 6. Constraints & Common Pitfalls

1. **No Mixed Element Coefficients as Operands**:
   Mixed element coefficients are not directly supported as operands. You must first use `ufl.split()` to split the mixed function into its individual components.
   * *Incorrect*: `FEMExternalOperator(u_mixed, ...)`
   * *Correct*: `u_1, u_2 = ufl.split(u_mixed)` $\rightarrow$ `FEMExternalOperator(u_1, u_2, ...)`
2. **Evaluation Order**:
   Always call `replace_external_operators` before `evaluate_operands` and `evaluate_external_operators`. The evaluation routines rely on metadata generated during the replacement step.
3. **Flattens**:
   The output of the implementation functions must be flattened (`.flatten()` or `.reshape(-1)`) before returning, to conform to the memory layouts expected by DOLFINx's interpolation layer.
4. **Mixed Space Outputs**:
   If the operator's target `function_space` is a mixed space, the values corresponding to all subspaces must be concatenated appropriately. Refer to the codebase tests (e.g., `test_mixed_cg_dg_space` in `test/test_external_operators_evaluation.py`) for the layout rules.

---

## 7. Developer Utilities

If you are modifying the repository or writing tests:
* **Linting & Formatting**:
  ```bash
  ruff check .
  ruff format .
  ```
* **Testing**:
  ```bash
  pytest -v test/
  ```
