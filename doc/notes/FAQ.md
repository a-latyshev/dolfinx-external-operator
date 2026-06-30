# [Preview] External operators in FEnICSx - FAQ

## When to use external operators and when not? Check-list!

1. There is a certain variable $N$ in the variational formulation $F(N(u);v)$ that is **NOT** expressible via UFL, i.e. it cannot be easily expressed via analytical expressions and you have to use numerical algorithm to compute its values.
2. 