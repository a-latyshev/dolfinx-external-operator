## Feedback/TODO

- Use optional dependencies not requirements.txt
- Optional dependencies can include other optional dependencies
- Learn to use reflow commands to wrap lines automatically
- Add linting step
- Description in demos far too wordy and contain information that is
  unnecessary with respect to understanding how to *use* the library as a tool,
  c.f. understanding how the library *works*.
- Describing the implementation - the text should go near the usage of the
  function.
- docstrings unnecessary in demos, describe the functions in Markdown cell.
- It is OK for the code and paper to have different authors (I removed Jeremy
  and Corrado, if they add code they can add their name back in).
- Always CamelCase for classes and lower_case for functions and variables.
- Typing is easier to read with e.g. ufl.Form vs Form.
- Similarly to the above, code is easier to read with more fully typed names
  e.g. ufl.Action, rather than using from ufl import Action.
- Regarding the non-linear heat demo, with JAX and numba the user has to learn
  three new concepts. It would be best to do the first demo with pure numpy,
  then have two short additional scripts showing how to implement the same
  thing with numba and JAX.

## Improvements

Several improvements can be made:
1. How to update operands only once?
2. Create an ancestor for all derivatives?
3. Allocate memory once for all operands?
4. Inherit from both ufl.ExternalOperator and fem.Function?
5. IDEA: Let the femExOp derivatives inherit evaluated operands and then
provide a map between external operators and their evaluated operands.
Write a more general algorithm that will distinguish "families" of external
operators in order to evaluate operands of a "family" only once.
6. Tests:
    1. Evaluation of an external operator that depends on another external
    operator.
    2. Several external operators in one form.
    3. Combination of the previous tests.
7. Provide operands with an appropriate functional space, where they are
   evaluated.
8. If the derivation does not chage the shape, use the same functional
space.

