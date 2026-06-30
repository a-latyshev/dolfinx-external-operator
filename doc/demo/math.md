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
# [Preview] Formalizing notation for External Operators

This document formalizes the mathematical notation and discrete tensor calculus required to work with external operators in FEniCSx. It bridges directional derivatives with complex variational formulations, particularly those involving multiple arguments, nested compositions, high-order tensors, and complex functional spaces (e.g., mixed elements). Ultimately, this notation aims to help users systematically formalize, linearize, and understand variational problems involving external operators.

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

We call this derivate as Jacobian of $F$ and denote:

$$
    J(\u ; \hu, v) := D_\u[\F(\u ; v)]\{ \hu \},
$$

where semicolon ($;$) separates non-linear (on the left) and linear (on the
right) operands of $J$ (similar to $F$).

```{note} 
Although this notation may look overloaded, in the presence of functionals with multiple arguments and compositions (see the chain rule below), it makes clear expressing derivatives of complex variational forms (see examples below).
```

### Chain rule

If $F$ depends on an intermediate operator $N(u)$ (a.k.a _external operator_), the derivative expands via the **chain
rule**. 

$$
    D_u[F \circ N(u)]\{ \hat{u} \} = D_u[F(N(u); v)]\{ \hat{u} \} = D_N[F]\{
    D_u[N] \{ \hat{u} \} \}
$$

Then the Jacobian can be defined as 

$$
    J(u ; \hat{N}, v) = D_N[F] \{ \hat{N} \},
$$
where the direction $\hat{N} = D_u[N] \{ \hat{u} \}$ is itself a directional
derivative of the operator $N$ in the direction $\hat{u}$.

If the functional $F$ depends on multiple nonlinear operands, we can distinguish
_total_ ($D_\cdot[\cdot]\{ \cdot \}$) and _partial_ Gâteaux ($\partial_\cdot[\cdot]\{ \cdot \}$) derivatives to make it clear, with respect to what
argument we take a directional derivative. For example,

$$
    D_u[F(u, w, N(u); v)]\{ \hat{u} \} = \partial_u [F(u, w, N(u); v)]\{ \hat{u}
    \} + \partial_N [F(u, w, N(u); v)]\{ \hat{N} \}, 
$$
where $\hat{N} = D_u[N] \{ \hat{u} \}$.

## The Derivative of the External Operator

The external operator $N$ may itself depend on an operand $o(u)$, thus we apply
the chain rule here again:

$$
    D_u[N(o(u))]\{ \hat{u} \} = \frac{\partial N}{\partial o} \cdot
    D_u [o(u)]\{ \hat{u} \}.
$$

Thus, the Gâteaux derivative of $N$ consists of two parts: 
1. The partial derivative of $N$ with respect to its operand: $\frac{\partial
   N}{\partial o}$. This part can be treated as a "normal" derivative of $N$. **This is
   exactly, what we mean by "derivative of external operator", which is a new
   external operator and thus has to be provided by user in a form of a program
   that computes values of $\frac{\partial
   N}{\partial o}$**.
2. The second part represents the derivative of an expression that depends on
   the main field $u$. This is typically a UFL expression like $\nabla u$. This
   part is handled by UFL automatically, except the case, when the operand
   $o(u)$ is another external operator.

External operator may, of course, depend on multiple operands (e.g. $o_1, o_2$ and $o_3$), which may depend on multiple main fields (e.g. $u, w$):

$$
    D_u[N(o_1(u), o_2(w), o_3(u))]\{ \hat{u} \} = \frac{\partial N}{\partial o_1} \cdot
    D_u [o_1(u)]\{ \hat{u} \} + \frac{\partial N}{\partial o_3} \cdot
    D_u [o_1(u)]\{ \hat{u} \}.
$$

### The General Tensor Case

To ensure dimensional consistency and generalize the external operator framework, we must define the tensor ranks and contraction rules explicitly. 

Let the external operator $\boldsymbol{N}(\boldsymbol{o}(u))$ be a tensor
function of rank $k$. Let its input argument, the operator $\boldsymbol{o}(u)$,
be a tensor of rank $p$, then the derivative of the external operator (a.k.a the tangent operator $\mathbb{C}$)

$$
    \mathbb{C} := \frac{\partial \boldsymbol{N}}{\partial \boldsymbol{o}}
$$

is a higher-order tensor of rank $k + p$.

When the external operator is varied in the direction $\hat{u}$, the resulting
directional derivative is a contraction of tensors

$$
    D_u[\boldsymbol{N}(\boldsymbol{o}(u))]\{ \hat{u} \} = \mathbb{C} : D_u [\boldsymbol{o}(u)] \{ \hat{u} \}
$$

#### Explicit Index Notation

Using the Einstein summation convention, let $\alpha_1 \dots \alpha_k$ denote the indices of the external operator $\boldsymbol{N}$, and $\beta_1 \dots \beta_p$ denote the indices of the kinematic operator $\boldsymbol{o}$.

The components of the tangent tensor $\mathbb{C}$ are:

$$\mathbb{C}_{\alpha_1 \dots \alpha_k \beta_1 \dots \beta_p} = \frac{\partial \boldsymbol{N}_{\alpha_1 \dots \alpha_k}}{\partial \boldsymbol{o}_{\beta_1 \dots \beta_p}}$$

The directional variation of the external operator is evaluated by contracting the tangent tensor $\mathbb{C}$ with the directional variation of operand $\bo$ over the $p$ indices of the latter:

$$
    \left(D_u[\boldsymbol{N}(\boldsymbol{o}(u))]\{ \hat{u} \}\right)_{\alpha_1 \dots \alpha_k} = \mathbb{C}_{\alpha_1 \dots \alpha_k \beta_1 \dots \beta_p} : (D_u [\boldsymbol{o}(u)] \{ \hat{u} \})_{\beta_1 \dots \beta_p}
$$

## Mixed-element external operators

Let's assume that the external operator $\boldsymbol{N}$ is from a mixed element
space $V$ which consists of $m$ subspaces:

$$
    V = V_1 \times \cdots \times V_m,
$$

so the external operator is represented as a block object

$$
    \boldsymbol{N} = (\boldsymbol{N}_1, \dots, \boldsymbol{N}_m),
$$

where each "sub" external operator is from the corresponding subspace:
$\boldsymbol{N}_i \in V_i$. 

```{note}
Although, there are independent "sub" external operators their values are being stored in one singled flattened vector preserving the order with respect to the sub functional space. See examples in `test_mixed_element_space`: https://github.com/a-latyshev/dolfinx-external-operator/blob/main/test/test_external_operators_evaluation.py#L185.
```

If the external operator has multiple operands $\boldsymbol{N} =
\boldsymbol{N}(\boldsymbol{o}_1, \dots, \boldsymbol{o}_n)$ then its
derivative may require allocation of new functional spaces due to the change of the mathematical shape. For example, let's consider the partial derivative of $\bN$ with respect to a certain operand $\bo_j$:

$$
    \frac{\partial \boldsymbol{N}}{\partial \boldsymbol{o_j}} = \left( 
        \frac{\partial \boldsymbol{N}_1}{\partial \boldsymbol{o_j}}, \dots, \frac{\partial \boldsymbol{N}_m}{\partial \boldsymbol{o_j}}
    \right), \quad  j = 1, \dots,n.
$$

If $\boldsymbol{N}_i$ is a tensor of rank $k_i$ and the operand
$\boldsymbol{o_j}$ has the rank $p_j$ then $\mathrm{rank}\left( \frac{\partial
\boldsymbol{N}_i}{\partial \boldsymbol{o_j}}\right) = k_i + p_j$. These new
derivatives $\frac{\partial
\boldsymbol{N}_i}{\partial \boldsymbol{o_j}}$ have to live in new functional spaces $Q_{ij}$ with appropriate dimensions.
`dolfinx-external-operator` is designed to create automatically the final mixed-element space $Q_{j}$ composted of 
$Q_{ij}$, where the derivative $\frac{\partial \boldsymbol{N}}{\partial \boldsymbol{o_j}}$ will live:

$$
    \frac{\partial \boldsymbol{N}}{\partial \boldsymbol{o_j}} \in Q_j = Q_{1j} \times \cdots \times Q_{mj}, \quad  j = 1, \dots,n.
$$

```{important}
Creation of a functional space is an expensive operation. When external operators and its operands are high-order tensors, a lot of memory may be allocated automatically.
```

**Example:**

Let's suppose that the external operator $\bN$ is from the mixed functional
space $\bV = V_1 \times \bV_2$, consisting of two subspaces: a scalar space
$V_1$ and a vector space $\bV_2$ in $\mathbb{R}^2$. Furthermore, $\bN$ depends
on two operands $o_1(\cdot) \in V_1$ and $\bo_2(\cdot) \in \bV_2$:

$$ \bN(o_1, \bo_2) = [N_1(o_1, \bo_2), \bN_2(o_1, \bo_2)]^T \in \bV = V_1 \times \bV_2$$

Here for simplicity we consider the following operands:

$$
    o_1 = o_1(u_1), \quad \bo_2 = \bo_2(\bu_2)
$$

For $\bu, \bv \in \bV$

$$
F(N_1, \bN_2;\bv) = \int\limits_\Omega N_1 v_1 + \bN_2 : \bv_2 \, \mathrm{d}\bx
$$

For the trial function $\hat{\bu} = [\hat{u}_1, \hat{\bu}_2]^T \in \bV$ from the mixed element space the directional derivative looks like follows

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

And finally, we expand the directional derivatives of the component of the external operator $\bN$.

$$
    D_{u_1}[N_1]\{ \hat{u}_1 \} = \frac{\partial N_1}{\partial o_1} \cdot
    D_{u_1} [o_1]\{ \hat{u}_1 \} + \frac{\partial N_1}{\partial \bo_2} :
    \underbrace{D_{u_1} [\bo_2]\{ \hat{u}_1 \}}_{=\bzero},
$$

$$
    D_{u_1}[\bN_2]\{ \hat{u}_1 \} = \frac{\partial \bN_2}{\partial o_1} \cdot
    D_{u_1} [o_1]\{ \hat{u}_1 \} + \frac{\partial \bN_2}{\partial \bo_2} :
    \underbrace{D_{u_1} [\bo_2]\{ \hat{u}_1 \}}_{=\bzero},
$$

$$
    D_{\bu_2}[N_1]\{ \hat{\bu}_2 \} = \frac{\partial N_1}{\partial o_1} \cdot
    \underbrace{D_{\bu_2} [o_1]\{ \hat{\bu}_2 \}}_{=0} + \frac{\partial N_1}{\partial \bo_2} :
    D_{\bu_2} [\bo_2]\{ \hat{\bu}_2 \},
$$

$$
    D_{\bu_2}[\bN_2]\{ \hat{\bu}_2 \} = \frac{\partial \bN_2}{\partial o_1} \cdot
    \underbrace{D_{\bu_2} [o_1]\{ \hat{\bu}_2 \}}_{=0} + \frac{\partial \bN_2}{\partial \bo_2} :
    D_{\bu_2} [\bo_2]\{ \hat{\bu}_2 \},
$$

While all directional derivatives of the operands $D_{\bu_i} [\bo_j]\{ \hat{\bu}_i  \}$ are handled automatically by UFL, the partial derivatives of $\bN$ with respect to its operands have to be provided by user explicitly. That's why, in practice, we just need to know the partial derivatives of the external operators:

$$
\frac{\partial \bN}{\partial o_1} = \left(\frac{\partial N_1}{\partial o_1}, \frac{\partial \bN_2}{\partial o_1} \right),
$$

$$
\frac{\partial \bN}{\partial \bo_2} = \left( \frac{\partial N_1}{\partial \bo_2}, \frac{\partial \bN_2}{\partial \bo_2} \right).
$$

As mentioned previously, in the context of mixed elements, we work with block objects, which means that the values of, e.g. $\frac{\partial N_1}{\partial o_1}$ and $\frac{\partial \bN_2}{\partial o_1}$ must be stored as a single contiguous flattened array preserving the order of the blocks. See examples in `test_mixed_element_space`: https://github.com/a-latyshev/dolfinx-external-operator/blob/main/test/test_external_operators_evaluation.py#L185.

After UFL differentiation of the form $F$, a new mixed element space $\bQ_2 = \bQ_{12} \times \bQ_{22}$ will be allocated, with the following mathematical shapes

$$
\mathrm{shape}(\bq_{12}) = (2), \quad \bq_{12} \in \bQ_{12},\\
\mathrm{shape}(\bq_{22}) = (2,2), \quad \bq_{22} \in \bQ_{22}.\\
$$

```{important}
Since the operand $o_1$ is a scalar, the differentiation with respect to $o_1$ doesn't change the mathematical shape. Therefore, there is no need in allocation of a new functional space $\bQ_{1}$ and we can simply reuse $\bV$ for $\frac{\partial \bN}{\partial o_1}$. This behavior may change in future releases of `dolfinx-external-operator`, when it will be required to decrease the polynomial degree for the space where the derivative of the external operator will live.
```