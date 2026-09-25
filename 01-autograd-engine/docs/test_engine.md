# Test Engine: Design Rationale

## Why Numerical Gradient Checking?

Autograd engines are notoriously subtle to get right. A single sign error in a _backward function
produces silently wrong gradients that still "look reasonable" during training -- the model coverges to a wrong solution.

Numeral gradients provide a mathematical ground truth derived from the definition of the derivate, independent of the autograd implementation. If analytical and numerical gradients math, the backward pass is correct

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

## Why Fres Value Objects Each Call?

```python
plus = [Value(d) for d in datas] #New graph
minus = [Value(d) for d in datas] #New graph
```

**Graph contamination.**

If we reused the same `Value` objects and mutated `.data`, the computational graph would accumulate multiple forward passes. The backward pass would then compute gradients for a Frankenstein graph that never existed in the forward direction.

Fresh objects guarantee each numerical gradient evaluation is a **clean, independent forward pass**
-- exactly matching the mathematical definition of a partial derivative.

## Why Perturb `.data` Directly Instead of `Value(d + eps)?

```python
plus[idx].data +=eps #Mutate existing node
```

**Isolating the variable of differentiation.**

Creating `Value(d + eps)` would work mathematically, but it obscures intent: we're not evaluating at nearby point, we're measuring sensitivity to an *infinitesimal cahnge in one specific input.*
Mutating `.data` on a pre-constructed graph makes the "wich variable" question explicit in the code structure.

---

## Why eps = 1e-6 as Default?

**The floating-point sweet spot.**

```
Too large (h > 1e-3):  Truncation error dominates (O(h²) term)
Too small (h < 1e-8):  Roundoff error dominates (subtraction cancellation)
Just right (h ≈ 1e-6): Both errors ~1e-12, near machine epsilon
```

## Why Not Test Agaist PyTorch/ JAX?

If we validate against PyTorch, we're assuming PyTorch is correct. That's reasonable for production, but this is an educational engine --the point is to verify our implementation from first principles.

Numerical gradients are the "first principles" baseline. Pythorch agreement is a sanity check, not a proof.

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
