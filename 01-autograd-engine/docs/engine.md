# Phase 1: Autograde Engine (`engine.py`)

## Why a `Value` class at all?

Python's `float` is a dead number. Once you compute `c = a + b`, the result knows nothing about `a` or `b`. It cannot tell you where it came from or how it would change if you nudged its inputs.

A neural network needs exactly that ability. Training is the process of asking *"if I nudge this weight by a tiny amount, how much does the final error change?"* A plain float cannot answer that. So we wrap every number in a `Value`, which remembers:

- **Its data** — the actual number.
- **Its parents** — which other `Value`s were combined to produce it.
- **The operation** — how they were combined.
- **Its gradient** — how much it affects the final output.

The result is a **node in a computational graph**. Every arithmetic operation adds new nodes and new edges. When you call `backward()` at the end, the graph answers the question above for every node at once.


## Core Functions 

### `__init__` (The memory)

The first version of this code used `set(_children)`. It looks cleaner: no duplicates, fast lookups. It is also **wrong** for one specific case.

Consider `a * a`. With a set, `(a, a)` collapses to `{a}` — a single parent. The backward function still works, because `_backward` adds the gradient to `a.grad` twice, both `+=` accumulate correctly. But the **graph** loses an edge. When you draw it, you see one arrow instead of two. The picture lies about the computation.

A tuple preserves both connections. It costs nothing in this codebase, and it makes the visualization honest. When the graph and the math disagree, the graph is what you will believe at 2 AM while debugging.

### `__add__` & `__mul__` (The Operations)

They perform standard addition and multiplication, but with a twist: they record the operation and link new result to its original inputs.
This builds the graph dynamically.

### `_backward` (The Time Machine)
This function travels backward through the recorded receipts. It calculates the **gradient** (using the Chain Rule), which is simply a compass
that tells the neural network: *"Should this parameter go up or down to reduce the error?*
