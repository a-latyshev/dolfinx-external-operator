# External operators and functional analysis

# Formalizing External Operators in Variational Forms

This document formalizes notation around the use of external operators in
FEniCSx. It bridges the continuous formulation of directional derivatives with
the discrete tensor calculus required when constitutive laws or source terms are
evaluated with external operators. We believe it facilitates formalizing
variational problems when complex high-order tensors and functional spaces (e.g.
mixed elements) are used.

## 1. The Directional Derivative of a Variational Form

In standard finite element analysis, the weak form of a nonlinear boundary value problem is expressed as finding $u \in V$ such that for all test functions $v \in \hat{V}$:

$$
    F(u; v) = 0
$$

To solve this nonlinear system using Newton-like methods, we must linearize the form. This requires computing the Gâteaux derivative (or directional derivative) of the functional $F$ at the current state $u$ in the direction of an incremental field $\hat{u} \in V$:

$$
    D_u[F]\{ \hat{u} \} = 
        \lim_{\epsilon \to 0} \frac{F(u + \epsilon \hat{u};v) - F(u;v) }{\epsilon}.
$$

We call this derivate as Jacobian of $F$ and denote:
$$
    J(u ; \hat{u}, v) := D_u[F(u ; v)]\{ \hat{u} \},
$$
where semicolon separates non-linear (on the left) and linear (on the right)
operands of $J$ (similar to $F$).

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
where the direction $\hat{N} = D_u[N] \{ \hat{u} \}$ is the directional
derivative of the operator $N$ in the direction $\hat{u}$.

If the functional $F$ depends on multiple nonlinear operands, we can distinguish
_total_ ($D_\cdot[\cdot]$) and _partial_ Gâteaux ($\partial_\cdot[\cdot]$) derivatives to make it clear, with respect to what
argument we take a directional derivative

$$
    D_u[F(u, w, N(u); v)]\{ \hat{u} \} = \partial_u [F(u, w, N(u); v)]\{ \hat{u}
    \} + \partial_N [F(u, w, N(u); v)]\{ \hat{N} \}, 
$$
where $\hat{N} = D_u[N] \{ \hat{u} \}$.

## 3. The Derivative of the External Operator

The external operator $N$ may itself depend on an operand $o(u)$, thus we apply
the chain rule here again but this time we don't work with functional:
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

External operator may, of course, depend on multiple operands:
$$
    D_u[N(o_1(u), o_2(w), o_3(u))]\{ \hat{u} \} = \frac{\partial N}{\partial o_1} \cdot
    D_u [o_1(u)]\{ \hat{u} \} + \frac{\partial N}{\partial o_3} \cdot
    D_u [o_1(u)]\{ \hat{u} \}.
$$

### 4. The General Tensor Case

To ensure dimensional consistency and generalize the external operator framework, we must define the tensor ranks and contraction rules explicitly. 

Let the external operator $\boldsymbol{N}(\boldsymbol{o}(u))$ be a tensor
function of rank $k$. Let its input argument, the operator $\boldsymbol{o}(u)$,
be a tensor of rank $p$, then the derivative of the external operator (or
tangent operator)

$$
    \boldsymbol{C} := \frac{\partial \boldsymbol{N}}{\partial \boldsymbol{o}}
$$

is a higher-order tensor of rank $k + p$.

When the external operator is varied in the direction $\hat{u}$, the resulting
directional derivative is a contraction of tensors

$$
    D_u[\boldsymbol{N}(\boldsymbol{o}(u))]\{ \hat{u} \} = \boldsymbol{C} : D_u [\boldsymbol{o}(u)] \{ \hat{u} \}
$$

### Explicit Index Notation

Using the Einstein summation convention, let $\alpha_1 \dots \alpha_k$ denote the indices of the external operator $\boldsymbol{N}$, and $\beta_1 \dots \beta_p$ denote the indices of the kinematic operator $\boldsymbol{o}$.

The components of the tangent tensor $\boldsymbol{C}$ are:

$$\boldsymbol{C}_{\alpha_1 \dots \alpha_k \beta_1 \dots \beta_p} = \frac{\partial \boldsymbol{N}_{\alpha_1 \dots \alpha_k}}{\partial \boldsymbol{o}_{\beta_1 \dots \beta_p}}$$

The variation of the external operator is evaluated by contracting the tangent tensor with the variation of the kinematic operator over the $p$ indices of $\boldsymbol{o}$:

$$
    \left(D_u[\boldsymbol{N}(\boldsymbol{o}(u))]\{ \hat{u} \}\right)_{\alpha_1 \dots \alpha_k} = \boldsymbol{C}_{\alpha_1 \dots \alpha_k \beta_1 \dots \beta_p} : (D_u [\boldsymbol{o}(u)] \{ \hat{u} \})_{\beta_1 \dots \beta_p}
$$


### Substituting back into the Variational Form

Returning to the full directional derivative of the variational form:

$$DF(u; v)[\delta u] = \int_{\Omega} \left( \boldsymbol{C} : Do(u)[\delta u] \right) : \mathbf{M}(v) \, dx$$

In an implementation (such as `dolfinx-external-operator`), the framework expects the user to provide two black-box functions evaluated at quadrature points:
1.  The residual contribution: $\boldsymbol{N}(o(u))$
2.  The tangent operator: $\mathbb{C}(o(u))$

By formalizing the separation of the rank-$(k+p)$ external tangent tensor $\mathbb{C}$ from the rank-$p$ symbolic variations $Do(u)[\delta u]$ and $\mathbf{M}(v)$, the framework successfully constructs the exact Newton system for arbitrary external tensor functions without requiring symbolic transparency into $\boldsymbol{N}$.
