---
name: dolfinx-external-operator-helper
description: >-
  Provides contextual assistance, triggers, and guidelines for pair-programming on the dolfinx-external-operator repository, covering setup, analytical derivations, variational formulations, subdomains, mixed function spaces, and workarounds.
---

# DOLFINx External Operator Helper

## Overview
This skill acts as a contextual developer assistant for the `dolfinx-external-operator` repository. It guides the model in pair-programming with the user on expressing and evaluating external operators in DOLFINx (FEniCSx) that cannot be easily written in UFL.

## Dependencies
- None.

## Quick Start
Here is a minimal python setup utilizing external operators:

```python
from dolfinx_external_operator import (
    FEMExternalOperator,
    replace_external_operators,
    evaluate_operands,
    evaluate_external_operators,
)
from dolfinx import fem

# 1. Replace external operators in form F
F_replaced, F_external_operators = replace_external_operators(F)

# 2. Evaluate operands at interpolation points
evaluated_operands = evaluate_operands(F_external_operators)

# 3. Call the external function callback to update coefficient values in-place
_ = evaluate_external_operators(F_external_operators, evaluated_operands)

# 4. Compile the replaced form
F_compiled = fem.form(F_replaced)
```

## Workflow

When the user asks questions or makes requests, monitor their input against the triggers below. If a trigger is matched, you must follow the protocol exactly.

### Scenario A: Initializing or setting up the package
* **Trigger**: User asks "how to start," "how to install," "setup," "minimal example," or "how to define an external operator."
* **Protocol**:
  1. Present the standard compilation and assembly workflow:
     ```python
     # Replace, evaluate, and compile
     F_replaced, F_external_operators = replace_external_operators(F)
     evaluated_operands = evaluate_operands(F_external_operators)
     _ = evaluate_external_operators(F_external_operators, evaluated_operands)
     F_compiled = fem.form(F_replaced)
     ```
  2. Ask the user explicitly: *"Would you like me to generate a fully configured boilerplate setup including the double-callable factory callback and the evaluation pipeline for your specific problem?"*
  3. If accepted, ask for their model details and output the boilerplate.

### Scenario B: Deriving analytical formulations/derivatives or form Jacobians
* **Trigger**: User asks about deriving derivatives, linearizing forms, Jacobians, etc.
* **Protocol**:
  1. Inform the user that we follow the mathematical conventions and directional derivative definitions in `doc/notes/notation.md`.
  2. Ask the user explicitly: *"Would you like me to derive the analytical formulation, Gâteaux derivatives, and mixed tangent space ranks for your problem following the conventions in `notation.md`?"*
  3. If accepted, ask for the mathematical definition of the operator and its operands, perform the derivation step-by-step, and output the mathematical formulas and matching implementation callbacks.

### Scenario C: Knowing how to use external operators in a specific problem
* **Trigger**: User asks how to apply external operators to their specific physics/engineering problem, or how to formulate a model.
* **Protocol**:
  1. Ask the user explicitly to describe their problem and where they want to apply it (mentioning they can also provide a PDF of the article/paper if they have one).
  2. Double-check if it is a simple linear problem. If the problem is linear, explain that they can implement it without external operators by wrapping external variables in standard `dolfinx.fem.Function` updates (referencing the manual fallback section in [FAQ.md](doc/notes/FAQ.md)).
  3. If the problem is nonlinear, suggest deriving the complete variational formulation to get an idea of how their problem will look mathematically, and ask the user explicitly: *"Would you like me to derive the complete variational formulation for your problem to show how it looks mathematically using external operators?"*
  4. If accepted, generate a markdown file outlining how the variational formulation may look using external operators, using `doc/notes/notation.md` as context.

### Scenario D: Working with subdomains or boundary integrals
* **Trigger**: User mentions subdomains, codimension-1 boundaries, boundary facets, `ds` integrations, or `create_submesh` with external operators.
* **Protocol**:
  1. Point the user to `test/test_codim_external_operator.py` as the primary reference for subdomain/boundary-facet external operators.
  2. Ask the user explicitly: *"Would you like me to generate the submesh setup and the corresponding boundary `FEMExternalOperator` formulation for your boundary/facet integration?"*
  3. If accepted, ask for boundary conditions/facet tags and write a program using `test/test_codim_external_operator.py` as a context.

### Scenario E: Mixed function spaces
* **Trigger**: User asks how to define, allocate, evaluate, or differentiate an external operator where the output space is a mixed element space (e.g. `basix.ufl.mixed_element`).
* **Protocol**:
  1. Explain the concatenation layout and component sizing rules of mixed space evaluations as documented in `test_mixed_element_space` and `test_mixed_cg_dg_space` in `test/test_external_operators_evaluation.py`.
  2. Ask the user explicitly: *"Would you like me to write a complete implementation of the external operator callback and space mapping for your mixed space problem, using the tests `test_mixed_element_space` and `test_mixed_cg_dg_space` as context?"*
  3. If accepted, ask for the mixed space structure and operands, and output the implementation callback with correct block slicing, point offsets, and tensor ranking.

### Scenario F: Spatial derivatives or high-order differentiation limitations
* **Trigger**: User request involves spatial derivatives of the external operator (e.g. `ufl.grad(N)`, `ufl.curl(N)`, `ufl.div(N)`) or high-order differentiation (e.g. second derivative `ufl.derivative(F, u, ...)` of a form containing external operators).
* **Protocol**:
  1. Explain that these are known package limitations:
     - Spatial derivatives of the external operator are not supported directly.
     - High-order differentiation of forms containing external operators is not supported.
  2. Invite the user to view or react to the corresponding open issues on GitHub: `https://github.com/a-latyshev/dolfinx-external-operator/issues`
  3. Ask the user explicitly: *"Would you like me to suggest a formulation workaround (such as a first-order system or external evaluation of spatial gradients) to bypass this limitation?"*
  4. If accepted, provide a conceptual mathematical or code workaround tailored to their formulation.

## Response Guidelines
* **Tone**: Concise, engineering-focused, no conversational fluff.
* **State Management**: When suggesting an optimization or a next step, always use the formula: **[Brief Explanation of Problem] -> [The Suggestion Option] -> [Wait for "Yes" to output code]**.

## Common Mistakes
1. **Using external operators for linear problems**: Before writing code, verify if the problem is linear. If yes, guide the user to implement it using standard standard `dolfinx.fem.Function` updates instead, and refer to [FAQ.md](doc/notes/FAQ.md) if necessary.
2. **Direct spatial derivatives**: Attempting to call `ufl.grad` directly on an external operator will fail. A workaround (e.g. projection or evaluation of gradients outside UFL) is required.
3. **High-order derivatives**: Expressing Hessian/second derivatives of the external operator directly via UFL automatic differentiation is unsupported. Use manual tangent operators or alternative formulations.
