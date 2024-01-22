# Simple example

Authors: Andrey Latyshev (University of Luxembourg, Sorbonne Université, andrey.latyshev@uni.lu)

In order to show how an external operator can be used in a variational setting in FEniCSx/DOLFINx, we want to start with a simple example. 

Let us denote an external operator that is not expressible through UFL by $N = N(u)$, where the function $u$ is its single operand from functional space $V$. To fix ideas, we consider the following linear form $F$:

$$
  F(N(u);v) = \int N(u)v \, dx
$$
More generally, we can think of $F(N, v)$ as a UFL-expressible (non)linear form of the function $N$. However, as a nonlinear expression of $u$, the form can not be expressed via UFL.

In a variational setting, we quite often need to compute the Jacobian of the Form $F$. In other words, we need to take the Gateau derivative of the functional $F$ in the direction of $\hat{u}$. Denoting the full and partial Gateau derivatives of a functional through $\frac{d }{d u}(\cdot)$ and $\frac{\partial}{\partial u}(\cdot)$ respectively, applying the chain rule and omitting the operand of $N$, we can express the Jacobian of $F$ as following:

$$
  J(N;\hat{u}, v) = \frac{dF}{du}(N;\hat{u}, v) = \frac{\partial F}{\partial N}(N; \frac{\partial N}{\partial u}(u;\hat{u}), v) = \int \hat{N}(u;\hat{u})v \, dx,
$$

where $\hat{N}(u;\hat{u}) = \frac{\partial N}{\partial u}(u;\hat{u})$ is a Gateau derivative of the external operator $N$ in  the direction of  $\hat{u}$.

Thus, the Jacobian $J$ involves the computation of $ \frac{\partial N}{\partial u}$ which can be seen as another external operator.

The behaviour of both external operators $N$ and $\frac{\partial N}{\partial u}$ must be defined by a user via any callable Python function.
