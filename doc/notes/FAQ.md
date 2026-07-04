# [Preview] External Operators in FEniCSx: FAQ

## Checklist: When should I use external operators?

You should consider using external operators if:
1. A variable $N$ in your variational form $F(N(u); v)$ **cannot be expressed in UFL** (e.g., it requires an external numerical algorithm to evaluate).
2. The problem is **nonlinear**, requiring the derivative of $F$, and you want to avoid manually deriving, implementing, and updating the derivatives of the external operator.

If your problem meets these criteria, `dolfinx-external-operator` will save you significant implementation time.

```{note}
If your external dependency is linear and simple, you might not need this package. You can instead wrap the external variables as standard `dolfinx.fem.Function` coefficients (see the next section).
```

(section-no-external-operators)=
## If I plan to use external software alongside FEniCSx, do I have to use external operators?

No. For simple problems, you can manually:
1. Create a function space and an associated `dolfinx.fem.Function` to represent your external variable.
2. Define a Python function that calls your external software to compute values.
3. Interpolate/project the operands onto the target function space and update the function's values.

For example, if you solve a linear problem where the residual $F$ depends on an external variable $N = N(u)$:

$$
F(N(u); v) = \int_\Omega N(u) v \, \mathrm{d}x
$$

You can implement it manually like this:

```python
S_element = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=1)
S = fem.functionspace(domain, S_element)
N = fem.Function(S, name="external_variable")

def N_eval(u: np.ndarray) -> np.ndarray:
    # Call external software...
    return N_values.reshape(-1)

F = N * v * ufl.dx

# Project u onto S to evaluate N
eval_points = S_element.interpolation_points
u_expr = fem.Expression(u, eval_points)

...

for i in iterations:
    # Inside the solution loop:
    u_eval = u_expr.eval(domain, cells)
    N.x.array[:] = N_eval(u_eval)
```

This manual approach works well for simple linear cases. However, if the problem is **nonlinear**, you must also manually compute the Jacobian:

$$
J(u; \hat{u}, v) = \int_\Omega N^\prime(u) \hat{u} v \, \mathrm{d}x
$$

This requires allocating another field for the derivative $N^\prime$ and implementing its update logic:

```python
dN = fem.Function(S, name="derivative_of_external_variable")

def dN_eval(u: np.ndarray) -> np.ndarray:
    # Call external software to evaluate derivative...
    return dN_values.reshape(-1)

J = dN * u_hat * v * ufl.dx
```

As the complexity grows — involving nested compositions, higher-order tensors, multiple fields, etc — managing function spaces, coefficients, Jacobians, and chain rule evaluations manually becomes tedious and error-prone. `dolfinx-external-operator` automates all of these steps.

```{seealso}
For a demonstration of this complexity, see the mixed-element example in [Some Notation for External Operators](./notation.md).
```

## Does the use of external operators add extra numerical overheads regarding the overall performance?

Not really.

As mentioned in the main article {cite:p}`latyshevExpressing2025`:
> For non-trivial constitutive models, the runtime of the user’s implementation of the external operator usually dominates the runtime of the other aspects of evaluating an external operator, in particular, the data transfer between DOLFINx and users implemented external operators. As discussed previously, this data transfer is performed by copying the values from one `ndarray` to another. Time spent on such a copy is only a small fraction with respect to the time taken to execute the user’s implementation of the operator. Notwithstanding this argument, to reach the highest level of performance we recommend users implemented external operators using just-in-time (JIT) compilation features available in libraries like Numba and JAX, or in a compiled language.

Generally speaking, if one wants to use external software with FEniCSx framework without `dolfinx-external-operator` (see {ref}`the previous section <section-no-external-operators>`), they will have to copy data from external software to the FEniCSx environment via the [`ndarray`-interface](https://numpy.org/doc/stable/reference/arrays.interface.html) **in any case**.

## I see that there are only tutorials on constitutive modelling in solid mechanics, so it cannot be applied to other branches of finite element simulations, right?

No. The tutorials focus on constitutive modeling because of the authors backgrounds. The package just facilitates the use of the external software within FEniCSx in general and does not depend on types of applications. It has a potential to be applied to any kind of application envolving the use of external software.

## How to get an access to external operators created automatically after `ufl.derivative`?

All external operators contained by any form can be extracted via `replace_external_operators`:
```python
J_replaced, J_external_operators = replace_external_operators(J_expanded)
```
Then you can look for specific external operators with a specific multi-index.
```python
for dex_op in J_external_operators:
    if dex_op.derivatives == (1,):
        print(dex_op)
        ...
```

## Access to external operator values

Every external operator is associated with a `dolfinx.fem.Function` coefficient. An access to the values of the external operator can be done through this coefficient via `ref_coefficient`:
```python
ex_op_values_numpy = ex_op.ref_coefficient.x.array
```

## Why does UFL need symbolic zero-simplification for derivatives? Can't the callback just return zero?

Suppose you have an external operator $N(u) = u$, which is linear. Its second derivative $N''(u)$ is mathematically zero. It is natural to ask: why can't the user's Python callback simply return a function of zeros for unsupported/zero higher-order derivatives? For example:
```python
def N_external(derivatives):
    if derivatives == (0,):
        return lambda u: u
    elif derivatives == (1,):
        return lambda u: np.ones_like(u)
    elif derivatives == (2,):
        return lambda u: np.zeros_like(u)  # returns zeros at runtime
```

While this runs, it is suboptimal and causes several compile-time and runtime issues:

1. **Assembly Overhead (Multiplication by Zero)**:
   If UFL does not know a derivative is symbolically `Zero` at compile-time, it compiles a C++ representation of the form that still includes the derivative operator. During assembly, DOLFINx will still execute the Python callback, allocate a coefficient `fem.Function`, evaluate all its operands at all quadrature points, and then perform cell-wise integration loops multiplying values by zero. For large 3D meshes, this adds significant computational overhead.

2. **Compilation Failures (Empty Forms)**:
   If a user takes the Gâteaux derivative of a bilinear form where some terms disappear, and the entire form evaluates to zero:
   - If the derivative is simplified to `Zero` symbolically, the entire form reduces to an empty UFL Form (`Form([])`). UFL and DOLFINx detect this and avoid compiling it.
   - If the derivative is *not* simplified symbolically, the form contains a `FEMExternalOperator` representing the derivative. When DOLFINx tries to compile it, it fails with a traceback containing:
     `ValueError: not enough values to unpack (expected 1, got 0)`
     because the compiler cannot find a valid integration domain for a form that only consists of zero-valued operators.

3. **Symbolic Graph Expansion**:
   Without a base case that simplifies to symbolic `Zero`, taking higher-order derivatives (e.g. for sensitivity analysis or Hessian-vector calculations) will lead to an infinitely growing symbolic expression tree, causing slow compile times and potential recursion limits.

To avoid this, `dolfinx-external-operator` supports **auto-detection** (`max_derivative_order="auto"`, which is the default). It probes the user's callback at initialization to detect the maximum supported derivative order. Any higher-order derivative is immediately simplified to a symbolic UFL `Zero` of the correct tensor shape at compile-time, pruning the expression graph before C++ code generation.


```{bibliography}
:filter: docname in docnames
```