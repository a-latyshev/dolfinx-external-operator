$$
\def\F{{\color{#ff7f0e} F}}
\def\u{{\color{#1f77b4} u}}
\def\hu{{\color{#2ca02c} \hat{u}}}
\def\bN{\boldsymbol{N}}
\def\bQ{\boldsymbol{Q}}
\def\bq{\boldsymbol{q}}
\def\bV{\boldsymbol{V}}
\def\bo{\boldsymbol{o}}
\def\bx{\boldsymbol{x}}
\def\bv{\boldsymbol{v}}
\def\bu{\boldsymbol{u}}
\def\bzero{\boldsymbol{0}}
$$
# Some Notation for External Operators

This document outlines the notation and discrete tensor calculus conventions
used when working with external operators in FEniCSx. It bridges directional
derivatives with complex variational formulations, particularly those involving
multiple arguments, nested compositions, high-order tensors, and mixed function
spaces. Ultimately, these conventions aim to help users systematically
formulate, linearize, and understand variational problems involving external
operators.

## The Directional Derivative of a Variational Form

In standard finite element analysis, the weak form of a nonlinear boundary value problem is expressed as finding $\u \in V$ such that for all test functions $v \in \hat{V}$:

$$
    \F(\u; v) = 0
$$

To solve this nonlinear system using Newton-like methods, we must linearize the form. This requires computing the Gâteaux derivative (or directional derivative) of the ${ \color{#ff7f0e}\text{functional }} \F$ at the ${ \color{#1f77b4}\text{argument }}\u$ in the ${ \color{#2ca02c}\text{direction }} \hu \in V$:

$$
    D_\u { [\F]} { \{ \hu \} }= 
        \lim_{\epsilon \to 0} \frac{\F(\u + \epsilon \hu; v) - \F(\u; v) }{\epsilon},
$$

We refer to this derivative as the Jacobian of $\F$ and denote it as:

$$
    J(\u ; \hu, v) := D_\u[\F(\u ; v)]\{ \hu \},
$$

where the semicolon ($;$) separates the non-linear (left) and linear (right) operands of $J$ (similar to $\F$).

```{note} 
Although this notation may look verbose, in the presence of functionals with multiple arguments and compositions (see the chain rule below), it clarifies the expression of derivatives for complex variational forms (see examples below).
```

### Chain rule

If $\F$ depends on an intermediate operator $N(\u)$ (also known as an *external operator*), the derivative expands via the **chain rule**:

$$
    D_\u[\F \circ N(\u)]\{ \hu \} = D_\u[\F(N(\u); v)]\{ \hu \} = D_N[\F]\{
    D_\u[N] \{ \hu \} \}
$$

Then the Jacobian can be defined as 

$$
    J(\u ; \hat{N}, v) = D_N[\F] \{ \hat{N} \},
$$
where the direction $\hat{N} = D_\u[N] \{ \hu \}$ is itself a directional
derivative of the operator $N$ in the direction $\hu$.

If the functional $\F$ depends on multiple nonlinear operands, we distinguish between
**total** ($D_\cdot[\cdot]\{ \cdot \}$) and **partial** Gâteaux ($\partial_\cdot[\cdot]\{ \cdot \}$) derivatives to clarify which argument the directional derivative is taken with respect to. For example:

$$
    D_\u[\F(\u, w, N(\u); v)]\{ \hu \} = \partial_\u [\F(\u, w, N(\u); v)]\{ \hu
    \} + \partial_N [\F(\u, w, N(\u); v)]\{ \hat{N} \}, 
$$
where $\hat{N} = D_\u[N] \{ \hu \}$.

## The Derivative of the External Operator

The external operator $N$ may itself depend on an operand $o(\u)$, thus we apply the chain rule again:

$$
    D_\u[N(o(\u))]\{ \hu \} = \frac{\partial N}{\partial o} \cdot
    D_\u [o(\u)]\{ \hu \}.
$$

Thus, the Gâteaux derivative of $N$ consists of two parts: 
1. The partial derivative of $N$ with respect to its operand: $\frac{\partial N}{\partial o}$. This part can be treated as a standard derivative of $N$. **This is exactly what we mean by the "derivative of an external operator", which is itself a new external operator and must be explicitly provided by the user in the form of a program that computes the values of $\frac{\partial N}{\partial o}$**.
2. The second part represents the derivative of an expression that depends on the main field $\u$ (typically a UFL expression like $\nabla \u$). This part is handled automatically by UFL, except when the operand $o(\u)$ is another external operator.

An external operator may, of course, depend on multiple operands (e.g., $o_1$, $o_2$, and $o_3$), which in turn may depend on multiple main fields (e.g., $\u$, $w$):

$$
    D_\u[N(o_1(\u), o_2(w), o_3(\u))]\{ \hu \} = \frac{\partial N}{\partial o_1} \cdot
    D_\u [o_1(\u)]\{ \hu \} + \frac{\partial N}{\partial o_3} \cdot
    D_\u [o_3(\u)]\{ \hu \}.
$$

### The General Tensor Case

To ensure dimensional consistency and generalize the external operator framework, we must define the tensor ranks and contraction rules explicitly. 

Let the external operator $\bN(\bo(\u))$ be a tensor function of rank $k$. Let its input argument, the operator $\bo(\u)$, be a tensor of rank $p$. Then, the derivative of the external operator (also known as the tangent operator $\mathbb{C}$)

$$
    \mathbb{C} := \frac{\partial \bN}{\partial \bo}
$$

is a higher-order tensor of rank $k + p$.

When the external operator is varied in the direction $\hu$, the resulting directional derivative is a contraction of tensors:

$$
    D_\u[\bN(\bo(\u))]\{ \hu \} = \mathbb{C} : D_\u [\bo(\u)] \{ \hu \}
$$

#### Explicit Index Notation

Using the Einstein summation convention, let $\alpha_1 \dots \alpha_k$ denote the indices of the external operator $\boldsymbol{N}$, and $\beta_1 \dots \beta_p$ denote the indices of the operand $\boldsymbol{o}$.

The components of the tangent tensor $\mathbb{C}$ are:

$$
\mathbb{C}_{\alpha_1 \dots \alpha_k \beta_1 \dots \beta_p} = \frac{\partial \boldsymbol{N}_{\alpha_1 \dots \alpha_k}}{\partial \boldsymbol{o}_{\beta_1 \dots \beta_p}}
$$

The directional variation of the external operator is evaluated by contracting the tangent tensor $\mathbb{C}$ with the directional variation of operand $\bo$ over the $p$ indices of the latter:

$$
    \left(D_u[\boldsymbol{N}(\boldsymbol{o}(u))]\{ \hat{u} \}\right)_{\alpha_1 \dots \alpha_k} = \mathbb{C}_{\alpha_1 \dots \alpha_k \beta_1 \dots \beta_p} : (D_u [\boldsymbol{o}(u)] \{ \hat{u} \})_{\beta_1 \dots \beta_p}
$$

## Mixed Element Formulations

Suppose we are working with a multi-field or multi-component system. The mathematical space consists of $m$ subspaces:

$$
    \bV = V_1 \times \cdots \times V_m,
$$

and we want to define external operators for each component:

$$
    N_i \in V_i, \quad i = 1, \dots, m,
$$

under a variational form:

$$
    F(N_1, \dots, N_m; \boldsymbol{v}) = 0,
$$

where $\boldsymbol{v} = (v_1, \dots, v_m)$ is the block test function. To linearize this form in Newton-like methods, we compute the Gâteaux derivative with respect to a trial function direction $\hat{\boldsymbol{u}} = (\hat{u}_1, \dots, \hat{u}_m) \in V$:

$$
    D_{\boldsymbol{u}}[F]\{\hat{\boldsymbol{u}}\} = \sum_{i=1}^m \partial_{N_i}[F]\{ D_{\boldsymbol{u}}[N_i]\{\hat{\boldsymbol{u}}\} \}.
$$

For each external operator component $N_i$, its variation relies on partial derivatives with respect to its operands $\boldsymbol{o}$:

$$
    D_{\boldsymbol{u}}[N_i]\{\hat{\boldsymbol{u}}\} = \sum_{j} \frac{\partial N_i}{\partial \boldsymbol{o}_j} \cdot D_{\boldsymbol{u}}[\boldsymbol{o}_j]\{\hat{\boldsymbol{u}}\}.
$$

While the variations of the operands $D_{\boldsymbol{u}}[\boldsymbol{o}_j]\{\hat{\boldsymbol{u}}\}$ are handled automatically by UFL, the tangent operators (partial derivatives $\frac{\partial N_i}{\partial \boldsymbol{o}_j}$) must be explicitly provided by the user.

If the external operator has multiple operands $\boldsymbol{N} = \boldsymbol{N}(\boldsymbol{o}_1, \dots, \boldsymbol{o}_n)$, its derivative may require the allocation of new function spaces due to the change of the mathematical shape. For example, let's consider the partial derivative of $\boldsymbol{N}$ with respect to a certain operand $\boldsymbol{o}_j$:

$$
\frac{\partial \boldsymbol{N}}{\partial \boldsymbol{o}_j} = \left( 
    \frac{\partial \boldsymbol{N}_1}{\partial \boldsymbol{o}_j}, \dots, \frac{\partial \boldsymbol{N}_m}{\partial \boldsymbol{o}_j}
\right), \quad  j = 1, \dots,n.
$$

If $\boldsymbol{N}_i$ is a tensor of rank $k_i$ and the operand $\boldsymbol{o}_j$ has the rank $p_j$, then $\mathrm{rank}\left(\frac{\partial \boldsymbol{N}_i}{\partial \boldsymbol{o}_j}\right) = k_i + p_j$. These new derivatives $\frac{\partial \boldsymbol{N}_i}{\partial \boldsymbol{o}_j}$ must live in new function spaces $Q_{ij}$ with appropriate dimensions.
`dolfinx-external-operator` is designed to automatically create the final mixed-element space $\bQ_j$ composed of $Q_{ij}$, where the derivative $\frac{\partial \boldsymbol{N}}{\partial \boldsymbol{o}_j}$ lives:

$$
\frac{\partial \boldsymbol{N}}{\partial \boldsymbol{o}_j} \in \bQ_j = Q_{1j} \times \cdots \times Q_{mj}, \quad  j = 1, \dots,n.
$$

```{important}
The creation of a function space is an expensive operation. When external operators and their operands are higher-order tensors, substantial memory may be allocated automatically.
```
**Example**

Let's suppose that the external operator $\bN$ is from the mixed function space $\bV = V_1 \times \bV_2$, consisting of two subspaces: a scalar space $V_1$ and a vector space $\bV_2$ in $\mathbb{R}^2$. Furthermore, $\bN$ depends on two operands $o_1(\cdot) \in V_1$ and $\bo_2(\cdot) \in \bV_2$:

$$
\bN(o_1, \bo_2) = (N_1(o_1, \bo_2), \bN_2(o_1, \bo_2)) \in \bV = V_1 \times \bV_2
$$

Here for simplicity we consider the following operands:

$$
o_1 = o_1(u_1), \quad \bo_2 = \bo_2(\bu_2)
$$

For $\bu, \bv \in \bV$

$$
F(N_1, \bN_2;\bv) = \int\limits_\Omega N_1 v_1 + \bN_2 \cdot \bv_2 \, \mathrm{d}\bx
$$

For the trial function $\hat{\bu} = (\hat{u}_1, \hat{\bu}_2) \in \bV$ from the mixed element space, the directional derivative is as follows:

$$
D_{\bu}[F]\{\hat{\bu}\} = D_{u_1}[F]\{ \hat{u}_1 \} + D_{\bu_2}[F] \{ \hat{\bu}_2 \},
$$

where, keeping in mind $F = F(N_1(o_1, \bo_2), \bN_2(o_1, \bo_2);\bv)$, we expand each term:

$$
D_{u_1}[F]\{ \hat{u}_1 \} = \partial_{N_1}[F]\{ D_{u_1}[N_1]\{ \hat{u}_1 \} \} + \partial_{\bN_2}[F]\{ D_{u_1}[\bN_2]\{ \hat{u}_1 \} \},
$$

$$
D_{\bu_2}[F]\{ \hat{\bu}_2 \} = \partial_{N_1}[F]\{ D_{\bu_2}[N_1]\{ \hat{\bu}_2 \} \} + \partial_{\bN_2}[F]\{ D_{\bu_2}[\bN_2]\{ \hat{\bu}_2 \} \}.
$$

And finally, we expand the directional derivatives of the component of the external operator $\bN$:

$$
D_{u_1}[N_1]\{ \hat{u}_1 \} = \frac{\partial N_1}{\partial o_1} D_{u_1} [o_1]\{ \hat{u}_1 \} + \frac{\partial N_1}{\partial \bo_2} \cdot \underbrace{D_{u_1} [\bo_2]\{ \hat{u}_1 \}}_{=\bzero},
$$

$$
D_{u_1}[\bN_2]\{ \hat{u}_1 \} = \frac{\partial \bN_2}{\partial o_1} D_{u_1} [o_1]\{ \hat{u}_1 \} + \frac{\partial \bN_2}{\partial \bo_2} \cdot \underbrace{D_{u_1} [\bo_2]\{ \hat{u}_1 \}}_{=\bzero},
$$

$$
D_{\bu_2}[N_1]\{ \hat{\bu}_2 \} = \frac{\partial N_1}{\partial o_1} \underbrace{D_{\bu_2} [o_1]\{ \hat{\bu}_2 \}}_{=0} + \frac{\partial N_1}{\partial \bo_2} \cdot D_{\bu_2} [\bo_2]\{ \hat{\bu}_2 \},
$$

$$
D_{\bu_2}[\bN_2]\{ \hat{\bu}_2 \} = \frac{\partial \bN_2}{\partial o_1} \underbrace{D_{\bu_2} [o_1]\{ \hat{\bu}_2 \}}_{=0} + \frac{\partial \bN_2}{\partial \bo_2} \cdot D_{\bu_2} [\bo_2]\{ \hat{\bu}_2 \},
$$

In practice, we just need to know the partial derivatives of the external
operators:

$$
\frac{\partial \bN}{\partial o_1} = \left(\frac{\partial N_1}{\partial o_1}, \frac{\partial \bN_2}{\partial o_1} \right),
$$

$$
\frac{\partial \bN}{\partial \bo_2} = \left( \frac{\partial N_1}{\partial \bo_2}, \frac{\partial \bN_2}{\partial \bo_2} \right).
$$

After UFL differentiation of the form $F$, a new mixed element space $\bQ_2 = \bQ_{12} \times \bQ_{22}$ will be allocated, with the following mathematical shapes:

$$
\begin{aligned}
\mathrm{shape}(\bq_{12}) &= (2), \quad \bq_{12} \in \bQ_{12}, \\
\mathrm{shape}(\bq_{22}) &= (2,2), \quad \bq_{22} \in \bQ_{22}.
\end{aligned}
$$

```{important}
Since the operand $o_1$ is a scalar, differentiation with respect to $o_1$ does not change the mathematical shape. Therefore, there is no need to allocate a new function space $\bQ_{1}$, and we can simply reuse $\bV$ for $\frac{\partial \bN}{\partial o_1}$. 
```

In DOLFINx, there are two ways to represent and solve such mixed formulations. The main difference lies in whether the components are handled monolithically or independently, which directly impacts how operands and derivatives are structured:

`````{grid}
:gutter: 2

````{grid-item-card} 
:columns: 6

`basix.ufl.mixed_element`

In this monolithic approach, the mixed space is defined at the element level using a single function space $V$.

```python
V = fem.functionspace(domain, basix.ufl.mixed_element([Ve1, Ve2]))
```

**Operands**

The external operator is represented as a single block vector-valued object:

$$
\boldsymbol{N} = (N_1, \dots, N_m) \in V.
$$

Because $\boldsymbol{N}$ is a single monolithic operator, it shares a global set of operands:

$$
\boldsymbol{N} = \boldsymbol{N}(\boldsymbol{o}_1, \dots, \boldsymbol{o}_n).
$$

Every component $N_i$ mathematically receives the exact same set of operands, regardless of whether a particular component actually depends on all of them.

**Derivatives**

The tangent operators are grouped globally:

$$
\frac{\partial \boldsymbol{N}}{\partial \boldsymbol{o}_j} = \left( \frac{\partial N_1}{\partial \boldsymbol{o}_j}, \dots, \frac{\partial N_m}{\partial \boldsymbol{o}_j} \right)  \in \bQ_{j}.
$$

Their values must be stored in a single contiguous flattened array preserving the block order of the component subspaces.
````

````{grid-item-card} 
:columns: 6

`ufl.MixedFunctionSpace`

In this block-structured approach, each component space $V_i$ is defined separately, and coupled at the form level.

```python
V1 = fem.functionspace(domain, Ve1)
V2 = fem.functionspace(domain, Ve2)
W = ufl.MixedFunctionSpace(V1, V2)
```

**Operands**

Each component external operator is defined as a separate, self-contained operator:

$$
N_i \in V_i.
$$

Each operator behaves as an independent variable and maintains its own localized, independent set of operands:

$$
N_i = N_i(\boldsymbol{o}_{i, 1}, \dots, \boldsymbol{o}_{i, n_i}).
$$

Components $N_i$ do not need to share or align their operands, allowing each to be defined only with the arguments it needs.

**Derivatives**

The partial derivatives are computed and provided separately for each operator, similar to how it's done in the case of normal function spaces:

$$
\frac{\partial N_i}{\partial \boldsymbol{o}_{i, j}} \in Q_{i, j}.
$$

No global vector concatenation or alignment of operand sets across components is required.

````
`````

```{seealso}
Although the `ufl.MixedFunctionSpace` approach is more straightforward and allocates less memory, `dolfinx-external-operator` supports both approaches. See examples for both cases in [`test_external_operators_evaluation.py`](https://github.com/a-latyshev/dolfinx-external-operator/blob/main/test/test_external_operators_evaluation.py#L254-L677). 
```