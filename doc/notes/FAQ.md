# [Preview] External operators in FEniCSx - FAQ

## When to use external operators and when not? Check-list!

1. There is a certain variable $N$ in the variational formulation $F(N(u);v)$
   that is **NOT** expressible via UFL, i.e. it cannot be easily expressed via
   analytical expressions and you have to use numerical algorithm to compute its
   values.
2. When you deal with a nonlinear problem, which requires computing the
   derivative of $F$ and you don't want to explicitly define this derivative
   including.

If your problem is complex enough, those two points are sufficient to save some time on implementation of your problem by using `dolfinx-external-operator`.

```{note}
The first point is already enough to envole external operators if you don't want to do extra work around wrapping the external calls. But may be your problem is "simple" enough to wrap your externally defined variables via `fem.Function` coefficients. See the next section.
```

## If I plan to use external software alongside with FEniCSx, I have to use external operators, right?

No. Even if you use external software it doesn't mean that you have to
necessarily use external operators. If you problem is simple enough, then you
always can manually (i) create a functional space and the associated coefficient
`dolfinx.fem.Function` to wrap your external variable, (ii) define Python
function that calls external software to evaluate values of the external
variable, (iii) project the operands of the external variable onto the
functional space of the latter.

For example, you solve a linear problem, where the right part $F$ depends on a variable $N = N(u) \in S$, which is evaluated via external software. For $u,v \in V$, we have

$$
F(N(u);v) = \int N(u) v \, \mathrm{d}x
$$

```python
S_element = basix.ufl.quadrature_element(domain.topology.cell_name(), degree=1)
S = fem.functionspace(domain, S_element)

N = fem.Function(S, name="external variable")

def N_eval(u: np.ndarray) -> np.ndarray
    ... # call to external functions
    return N_values.reshape(-1)

F = N * v ulf.dx

# if `u` is defined in another functional space,
# we need to project it onto `S`, where `N` exists
eval_points = S_element.interpolation_points
u_expr = fem.Expression(u, eval_points) 

...

for i in iterations:
    u_eval = u_expr.eval(domain, cells) # evaluate u at quadratures
    N.x.array[:] = N_eval(u_eval) # update external variable values
    # solve problem
```

Wrapping anything that goes beyond UFL is simple: you just need a
`fem.Function`, which values are simple to update via NumPy arrays. 

Things become more complicated when one considers, for instance, nonlinearity. In this case, we should compute the Jacobian for the form $F$:

$$
J(u;\hat{u} , v) = \int N^\prime(u)\hat{u} v \, \mathrm{d}x
$$

This forces us to allocate an extra field that will store the values of $N^\prime$ and manually define $J$.

```python
dN = fem.Function(S, name="derivative of external variable")

def dN_eval(u: np.ndarray) -> np.ndarray
    ... # call to external functions
    return dN_values.reshape(-1)

J = dN * u_hat * v ulf.dx
```
s
Then we can further make this problem more complex by considering higher order
tensors, nesting compositions, multiple main fields, etc. More complex setups
will force us to write a lot of code manually: new functional spaces, new
cofficients, Jacobains, evaluation of operands... What
`dolfinx-external-operator` really does, it wraps all those steps automatically,
so the user don't need to even...

```{seealso}
Take a look at the example with the use of mixed elements on the page with [Some Notation for External Operators](./notation.md) for a "sufficiently" complex example to use the external operators. 
```

## Performance issues

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
To get an access to the values is done via `ref_coefficient`:
```python
values_numpy = dex_op.ref_coefficient.x.array
```

## I see that there are only tutorials on constitutive modelling in solid mechanics, so it cannot be applied to other branches of finite element simulations, right?

No. The tutorials focus on constitutive modeling because of the authors backgrounds. The package just facilitates the use of the external software within FEniCSx. It has a potential to be applied to any kind of application envolving the use of external software.