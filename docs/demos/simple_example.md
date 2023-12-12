# Simple example

Authors: Andrey Latyshev (University of Luxembourg, Sorbonne Université, andrey.latyshev@uni.lu)

In order to show how an external operator can be used in a variational setting in FEniCSx/DOLFINx, we want to start with a simple example. 

Let us denote an external operator that is not expressible through UFL by $N = N(u)$, where $u$ is its single operand from functional space $V$. In these terms, we consider the following linear form $F$.

$$
  F(N(u);v) = \int N(u)v \, dx
$$

In a variational setting, we quite often need to compute the Jacobian of the Form $F$. In other words, we need to take the Gateau derivative of the functional $F$ in the direction of $\hat{u}$. Denoting the full and partial Gateau derivatives of a functional through $\frac{d }{d u}(\cdot)$ and $\frac{\partial}{\partial u}(\cdot)$ respectively, applying the chain rule and omitting the operand of $N$, we can express the Jacobian of $F$ as following:

$$
  J(N;\hat{u}, v) = \frac{dF}{du}(N;\hat{u}, v) = \frac{\partial F}{\partial N}(N; \hat{N}, v) = \int \hat{N}(u;\hat{u})v \, dx,
$$

where $\hat{N}(u;\hat{u})$ is a new trial function, which behaviour is defined by the Gateau derivative $\frac{\partial N}{\partial u}(u;\hat{u})$.
___
or

$$
  J(N;\hat{u}, v) = \frac{dF}{du}(N;\hat{u}, v) = \frac{\partial F}{\partial N}(N; \hat{N}, v) \circ \left( \hat{N} = \frac{\partial N}{\partial u}(u;\hat{u}) \right) = \int \hat{N}(u;\hat{u})v \, dx,
$$

where $\hat{N}(u;\hat{u}) = \frac{\partial N}{\partial u}(u;\hat{u})$ is a Gateau derivative of the external operator $\hat{N}$.
___
or

$$
  J(N;\hat{u}, v) = \frac{dF}{du}(N;\hat{u}, v) = \frac{\partial F}{\partial N}(N; \frac{\partial N}{\partial u}(u;\hat{u}), v) = \int \hat{N}(u;\hat{u})v \, dx,
$$

where $\hat{N}(u;\hat{u}) = \frac{\partial N}{\partial u}(u;\hat{u})$ is a Gateau derivative of the external operator $\hat{N}$.
___
or

$$
  J(N;\hat{u},v) = F^\prime(N; \hat{N}(u;\hat{u}),v) = (F^\prime \circ \hat{N})(u;\hat{u},v)
$$

___

Chain rule (according to [wiki](https://en.wikipedia.org/wiki/Gateaux_derivative)):

\begin{align*}
  & H(u) = (G \circ F)(u) = G(F(u)) \\
  & H^\prime(u; \hat{u}) = (G \circ F)^\prime(u; \hat{u}) = G^\prime(F(u); \hat{F}(u; \hat{u})),
\end{align*}
where $\hat{G}(u; \hat{u}) = G^\prime(u; \hat{u})$.

May we write ?

$$
H^\prime(u; \hat{u}) = (G \circ F)^\prime(u; \hat{u}) = G^\prime(F(u); \hat{F}(u; \hat{u})) = (G^\prime \circ \hat{F})(u; \hat{u}),
$$
___

Thus, the Jacobian $J$ is presented as an action of the functional $\frac{\partial F}{\partial N}$ on the trial function $\hat{N}$....

The behaviour of both external operators $N$ and $\frac{d N}{d u}$ must be defined by a user via any callable Python function.
