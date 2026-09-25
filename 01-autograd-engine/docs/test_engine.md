# Test Engine: Design Rationale

This document explains the design decisions behind the numerical gradient checking framework used to verify the autograd engine. Each section covers a specific choice: why it was made, what alternatives were considered, and how it maps to the implementation.

## Why Numerical Gradient Checking?

Autograd engines are notoriously subtle to get right. A single sign error in a _backward function
produces silently wrong gradients that still "look reasonable" during training -- the model converges to a wrong solution.

Numerical gradients provide a mathematical ground truth derived from the definition of the derivative, independent of the autograd implementation. If analytical and numerical gradients match, the backward pass is correct.

---

## Why Central Difference Over Forward Difference?

Accuracy per function evaluation

| Method  | Evaluations | Error | Cost for 1 gradient |
| ------- | ----------- | ----- | ------------------- |
| Forward | 2           | O(h)  | 2                   |
| Central | 2           | O(h²) | 2                   |

Same computational cost, quadratically better accuracy. With h = 1e-6, central difference hits machine precision (~1e-12) while forward difference stalls at ~1e-6

**Why not higher-order methods?** (5-points stencil, Richardson extrapolation)
    - Diminishing returns: central difference already hits floating-point limits
    - More evaluations = slower tests
    - More complex = more surface for test bugs

---

## Why Fresh Value Objects Each Call?

```python
plus = [Value(d) for d in datas] # New graph
minus = [Value(d) for d in datas] # New graph
```

**Graph contamination.**

If we reused the same `Value` objects and mutated `.data`, the computational graph would accumulate multiple forward passes. The backward pass would then compute gradients for a Frankenstein graph that never existed in the forward direction.

Fresh objects guarantee each numerical gradient evaluation is a **clean, independent forward pass** — exactly matching the mathematical definition of a partial derivative.

## Why Perturb `.data` Directly Instead of `Value(d + eps)`?

```python
plus[idx].data += eps  # Mutate existing node
```

**Isolating the variable of differentiation.**

Creating `Value(d + eps)` would work mathematically, but it obscures intent: we're not evaluating at a nearby point, we're measuring sensitivity to an *infinitesimal change in one specific input.* Mutating `.data` on a pre-constructed graph makes the "which variable" question explicit in the code structure.

## Why eps = 1e-6 as Default?

**The floating-point sweet spot.**

```
Too large (h > 1e-3):  Truncation error dominates (O(h²) term)
Too small (h < 1e-8):  Roundoff error dominates (subtraction cancellation)
Just right (h ≈ 1e-6): Both errors ~1e-12, near machine epsilon
```

## Why Not Test Against PyTorch/JAX?

If we validate against PyTorch, we're assuming PyTorch is correct. That's reasonable for production, but this is an educational engine — the point is to verify our implementation from first principles.

Numerical gradients are the "first principles" baseline. PyTorch agreement is a sanity check, not a proof.

## Why Tolerance Tables Per Operation?

**Error propagation differs by operation.**

```
Addition:        Errors add linearly          → tight tolerance (1e-7)
Multiplication:  Errors scale with magnitude  → looser (1e-6)
Transcendental:  Multiple rounding steps      → loosest (1e-5)
Deep chains:     Error compounds per layer    → adaptive
```

A single global tolerance either:
- Too strict: False failures on legitimate floating-point noise
- Too loose: Misses actual bugs in sensitive operations

Per-operation tolerances reflect the **numerical conditioning** of each primitive.

---

## Why Test Edge Cases Separately?

**Gradients behave differently at boundaries.**

| Region | Typical Behavior | Failure Mode |
|--------|------------------|--------------|
| `x > 0` (ReLU) | Gradient = 1 | Never tested if only positive inputs |
| `x = 0` (ReLU) | Gradient = 0 (subgradient) | Wrong subgradient choice |
| `x < 0` (ReLU) | Gradient = 0 | Dead neuron never detected |
| `x ≈ 0` (1/x) | Gradient explodes | Numerical instability masked |

Interior points are "easy" — most implementations get them right. Boundaries reveal design decisions (subgradient at 0? NaN? 0?).

---

## Why This Import Dance?

```python
try:
    from engine import Value
except ImportError:
    sys.path.insert(0, parent_dir)
    from engine import Value
```

**Tests must run from anywhere.**

Developers run tests from:
- Project root (`pytest tests/`)
- Test directory (`python test_engine.py`)
- IDE "run test" buttons (unpredictable CWD)
- CI pipelines (controlled but distinct)

The try/except makes the test **location-agnostic** — a quality-of-life detail that prevents "works on my machine" friction.

---

## How `numeric_grad()` Works

**Arguments**

- `f` — the function whose derivative we want. It takes `Value` objects and returns a `Value`.
- `data` — a tuple/list of raw numbers. These are the *inputs* of `f`.
- `idx` — which input to differentiate with respect to. The derivative of `f` w.r.t `datas[idx]`.
- `eps` — the tiny step used for the finite difference. `1e-6` is a standard choice: small enough to approximate the limit, large enough that floating-point cancellation does not dominate.

### Why `datas` and `idx` Separately?

Because `f` can have many inputs (a + b, a * b, etc.). To build the gradient vector, we compute the partial derivative w.r.t each input one at a time, nudging only that one and keeping the others fixed. `idx` says which one is being nudged.

```python
plus = [Value(d) for d in datas]
```

We take every raw number in `datas` and wrap it in a fresh `Value`. Why fresh? Because if we reused the caller's `Value` objects, calling `f(*plus)` would build a graph on them, and any future `backward()` would accumulate gradients onto them. That would silently corrupt later tests.

Each call to `numeric_grad` gets its own isolated set of `Value`s, with no connection to anything else.

```python
plus[idx].data += eps
```

Here is where the actual finite difference happens. We take only the `idx`-th input and add `eps` to its `.data`, not to the `Value` object itself.

Why `.data` and not something else? Because `.data` is the raw number. Adding `eps` to it changes the numeric value of that input, which is exactly what a derivative asks: *"if this input changes by a tiny amount, how does the output change?"*

Note we mutate `.data` directly, in place. The `plus` list now represents the point (x₀, x₁, x₂, ..., xᵢ + ε, ...).

```python
f_plus = f(*plus).data
```

We call `f` with nudged values. `f` runs the forward pass and returns a `Value` (the output of the expression). We immediately read its `.data` — the raw number — because we only need the numeric result, not the graph.

`f(*plus)` unpacks the list into positional arguments. If `plus = [a,b]`, then `f(*plus)` is `f(a,b)`.

```python
minus = [Value(d) for d in datas]
minus[idx].data -= eps
f_minus = f(*minus).data
```

Exact mirror of the first three lines, but this time we subtract `eps` instead of adding it. We end up with `f_minus = f(x₀, x₁, x₂, ..., xᵢ - ε, ...)`.

Now we have two function evaluations: one at xᵢ + ε, one at xᵢ - ε. Everything else is identical.

### The Central Difference Formula

```python
return (f_plus - f_minus) / (2 * eps)
```

This is the **central difference** approximation of the derivative:

f'(x) ≈ (f(x + ε) − f(x − ε)) / (2ε)

---

## How `check()` Works

- `f` — the function to test
- `*datas` — any number of raw input numbers. `check(lambda a, b: a * b, 2.0, 3.0)` passes two inputs.
- `tol` — how much discrepancy is acceptable. `1e-5` is generous enough to absorb the approximation error of finite differences, but tight enough to catch real bugs (which are usually off by orders of magnitude, not by tiny fractions).

```python
for i in range(len(datas)):
    analytic = values[i].grad
    numeric = numeric_grad(f, datas, i)
```

We loop over each input. For each one:
- `analytic` — the gradient your engine computed, read from `.grad`.
- `numeric` — the gradient computed independently by finite differences.

If your `_backward` functions are correct, these two numbers are the same up to floating-point noise.

If the two numbers differ by more than `tol`, we raise `AssertionError` with a message that shows both values and which input failed.

The message is the most important part. When a test fails, this string is what you see. It tells you:

**Which input** (input 0, input 1, ...) had the wrong gradient.
**What your engine computed** (analytic=...)
**What the true value should be** (numeric=...)

---

## Why the `check()` Function?

Every test follows the same pattern:
- Build graph with `Value` wrappers
- Run `backward()`
- Compare each input's `.grad` vs `numeric_grad`

Without `check()`, each test duplicates 8-10 lines. With it, tests become one-liners.