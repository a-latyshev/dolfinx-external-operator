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

If $F$ depends on an intermediate operator $N(u)$ (such as a gradient $\nabla u$
or a strain tensor $\varepsilon(u)$), the derivative expands via the **chain
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

## 2. Introducing the External Operator

An **external operator** $\mathbf{N}$ arises when a component of the integrand cannot or should not be represented symbolically. This frequently occurs with complex material models, neural network-based constitutive laws, or legacy Fortran/C++ subroutines evaluated strictly at quadrature points.

Consider a modified variational form where the integrand involves an inner product with an external operator $\mathbf{N}$, which itself depends on a kinematic operator $o(u)$:

$$F(u; v) = \int_{\Omega} \mathbf{N}(o(u)) : \mathbf{M}(v) \, dx$$

Here, $\mathbf{M}(v)$ represents a test function operator (e.g., the symmetric gradient of the test function, $\nabla^s v$).

Taking the directional derivative of this form yields:

$$DF(u; v)[\delta u] = \int_{\Omega} D\mathbf{N}(o(u))[\delta u] : \mathbf{M}(v) \, dx$$

The term $D\mathbf{N}(o(u))[\delta u]$ is the variation of the external operator. Because $\mathbf{N}$ is a "black box" to the symbolic math engine, the software framework cannot automatically differentiate it with respect to $u$. The mathematical formulation must isolate the derivative of the external function from the variation of the kinematic field.

## 3. The Derivative of the External Operator

To compute $D\mathbf{N}(o(u))[\delta u]$, we decouple the variation of the primary field $u$ from the internal logic of $\mathbf{N}$. We apply the partial Gâteaux derivative framework defined by the chain rule:

$$D\mathbf{N}(o(u))[\delta u] = \frac{\partial \mathbf{N}}{\partial o} : Do(u)[\delta u]$$

This equation cleanly separates two responsibilities:

1.  **$Do(u)[\delta u]$ (Kinematic Variation):** The variation of the intermediate operator with respect to the primary field. For example, if $o(u) = \nabla u$, then $Do(u)[\delta u] = \nabla \delta u$. This is purely kinematic and is handled easily by the symbolic finite element framework.
2.  **$\frac{\partial \mathbf{N}}{\partial o}$ (The Tangent Operator):** The partial derivative of the external function with respect to its input tensor. This is the **consistent tangent** or algorithmic Jacobian. It contains the core physical or mathematical logic of the external model and must be provided by the external routine alongside the evaluation of $\mathbf{N}$ itself.

## 4. The General Tensor Case

To ensure dimensional consistency and generalize the external operator framework, we must define the tensor ranks and contraction rules explicitly. 

Let the external operator $\mathbf{N}$ be a tensor function of rank $k$. Let its input argument, the operator $o(u)$, be a tensor of rank $p$. 

The tangent operator provided by the external subroutine is defined as:

$$\mathbb{C} = \frac{\partial \mathbf{N}}{\partial o}$$

Because $\mathbf{N}$ is rank $k$ and $o$ is rank $p$, the tangent operator $\mathbb{C}$ is a higher-order tensor of rank $k + p$. 

When the external operator is varied in the direction $\delta u$, the resulting directional derivative is:

$$D\mathbf{N}(u)[\delta u] = \mathbb{C} : Do(u)[\delta u]$$

### Explicit Index Notation

Using the Einstein summation convention, let $\alpha_1 \dots \alpha_k$ denote the indices of the external operator $\mathbf{N}$, and $\beta_1 \dots \beta_p$ denote the indices of the kinematic operator $o$.

The components of the tangent tensor $\mathbb{C}$ are:

$$\mathbb{C}_{\alpha_1 \dots \alpha_k \beta_1 \dots \beta_p} = \frac{\partial \mathbf{N}_{\alpha_1 \dots \alpha_k}}{\partial o_{\beta_1 \dots \beta_p}}$$

The variation of the external operator is evaluated by contracting the tangent tensor with the variation of the kinematic operator over the $p$ indices of $o$:

$$\left( D\mathbf{N} \right)_{\alpha_1 \dots \alpha_k} = \mathbb{C}_{\alpha_1 \dots \alpha_k \beta_1 \dots \beta_p} \left( Do \right)_{\beta_1 \dots \beta_p}$$

### Substituting back into the Variational Form

Returning to the full directional derivative of the variational form:

$$DF(u; v)[\delta u] = \int_{\Omega} \left( \mathbb{C} : Do(u)[\delta u] \right) : \mathbf{M}(v) \, dx$$

In an implementation (such as `dolfinx-external-operator`), the framework expects the user to provide two black-box functions evaluated at quadrature points:
1.  The residual contribution: $\mathbf{N}(o(u))$
2.  The tangent operator: $\mathbb{C}(o(u))$

By formalizing the separation of the rank-$(k+p)$ external tangent tensor $\mathbb{C}$ from the rank-$p$ symbolic variations $Do(u)[\delta u]$ and $\mathbf{M}(v)$, the framework successfully constructs the exact Newton system for arbitrary external tensor functions without requiring symbolic transparency into $\mathbf{N}$.
