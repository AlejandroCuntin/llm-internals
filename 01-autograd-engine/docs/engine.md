# `engine.py` — Deep Dive

The README told you *what* `engine.py` does. This document tells you *why*.

---

## Why a `Value` class at all?

Python floats are dead numbers. Once you compute `c = a + b`, the result forgets where it came from. A neural network needs the opposite: for every number, it must know *"if I nudge this, how much does the final error change?"*

So we wrap every number in a `Value`, which remembers its data, its parents, the operation that created it, and its gradient. Every operation adds nodes and edges to a **computational graph**. Calling `backward()` at the end answers the nudge question for every node at once.

---

## Why a tuple for `_prev`, not a set

`a * a` with a set collapses to a single parent. The math still works (gradients accumulate via `+=`), but the drawn graph loses an edge and lies about the computation. A tuple preserves both connections.

---

## Why `_backward` is a closure

The local derivative often depends on the **values** of the inputs. For `a * b`, the derivative w.r.t. `a` is `b` — we need to remember `b` from forward time. Defining `_backward` inside the operation lets it capture `self`, `other`, and `out` from the enclosing scope. That is exactly what PyTorch does with its `grad_fn` objects.

---

## Why `+=` and not `=`

A node can feed multiple downstream paths. Each path contributes gradient, and the chain rule says these contributions **add up**. Using `=` would overwrite the first path with the second. Only graphs with fan-out reveal the bug — real networks always have fan-out.

Side effect: gradients accumulate between `backward()` calls. That is why `zero_grad()` exists.

---

## Why `__neg__`, `__sub__`, `__truediv__` have no backward

`a - b` is `a + (-b)`. `a / b` is `a * b**-1`. `-a` is `a * -1`. If the math is an identity, the code can be too. We get their gradients for free, and avoid four extra backward functions to debug.

---

## Why `__pow__` blocks negative bases with fractional exponents

`(-2.0) ** 0.5` in Python returns a **complex number** silently. Inside the graph, a complex `.data` would propagate through every operation, poisoning gradients and weights. You would only notice via unexplained `nan`s much later. The assertion fails fast with a clear and message.

---

## Why `tanh` and `relu` matter so much

**Stacking linear operations gives you another linear operation.** A 100-layer network of only `+` and `*` is mathematically a single matrix. Nonlinearities are the door through which curvature enters. Without them, depth is meaningless.

`tanh` is bounded and its derivative reuses its own output (`1 - tanh²`) — no recomputation needed. `relu` has derivative 1 or 0, which makes gradients cheap and sparse, enabling very deep networks to train.

---

## Why `exp` and `log` are in the engine

They are building blocks. `sigmoid`, `softmax`, and cross-entropy loss are all one line away once you have them. The overflow guard on `exp` (`x > 709`) turns a cryptic `OverflowError` from deep inside `math` into a clear message pointing at the offending value.

---

## Why `__radd__`, `__rmul__`, etc.

When Python evaluates `2 * x` where `x` is a `Value`, it calls `int.__mul__` first. The `int` does not know what a `Value` is, returns `NotImplemented`, and only then does Python try `x.__rmul__(2)`. Without these reflected methods, every `learning_rate * weight` would crash.

---

## Why `__eq__` compares by identity

Our `_topo` uses `visited = set()`. If `__eq__` compared by **value**, then two different `Value(0.0)` nodes would be treated as the same node. The set would skip the second one, its `_backward` would never be called, and gradients would be **silently wrong** — no error, just a network that trains badly for no visible reason.

So `Value == Value` is identity. `Value == 3` compares by value, because that is what users ask.

Then there is the second trap: defining `__eq__` makes Python set `__hash__ = None`, which would break our `set()` calls. We restore it with `__hash__ = object.__hash__`.

---

## Why `_topo` is iterative

Backprop requires every node's gradient to be **complete** before it is used. A topological sort gives the correct order. The iterative version (explicit stack) avoids Python's 1000-frame recursion limit, which a deep network would hit before finishing.

---

## Why `zero_grad` is not automatic

Accumulation is a feature: you can split a batch and call `backward()` several times to get the gradient of the full batch. Automatic reset would break that. But accumulation is also a footgun: forget to reset and your learning rate effectively explodes. So the reset is explicit, one line, non-negotiable.

---

## The core idea in one paragraph

Every `Value` is a node in a graph. Every operation adds nodes and edges. Every `_backward` is one local chain rule step. `backward` sorts the graph topologically, seeds the output with `1.0`, and calls each local rule in reverse. Gradients accumulate in `.grad` through `+=`. Everything else — `zero_grad`, `__hash__`, the tuple for `_prev`, the overflow guards — exists to keep that core loop correct under edge cases: shared nodes, negative bases, complex numbers, deep graphs, batched training.

**Local rules + topological order + accumulate.** Three ideas, one small engine.