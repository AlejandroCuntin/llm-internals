# `nn.py` — Deep Dive

`engine.py` gives you a machine that computes gradients for any expression. That is powerful, but it is also tedious: writing a network by hand means declaring hundreds of `Value` objects, wiring them together, and hoping you did not misplace an index.

`nn.py` is the layer that turns arithmetic into architecture. Its job is to make "a neural network" a single object you can build, call, and train.

This document explains why each class exists and what problem it solves. It follows the same principle as `engine.md`: the *what* is visible in the code, the *why* is here.

---

## Why separate `nn.py` from `engine.py`

A `Value` knows nothing about neurons. It knows how to add, multiply, and backpropagate. That is all it should ever know. If we started adding `w · x + b` logic into `engine.py`, three things would go wrong:

1. **The engine stops being reusable.** A `Value` is useful for any differentiable expression, not just neural networks. Baking in "this is a layer" would tie it to one use case.
2. **The engine gets harder to test.** Right now `test_engine.py` verifies gradients of pure math. That test should not need to know what a neuron is.
3. **The abstraction barrier blurs.** When something goes wrong in training, you want to know whether the bug is in the math or in the architecture. Keeping them in separate files makes that question trivial to answer.

`nn.py` imports `from engine import Value` and never the other way around. That direction is the contract: the engine is the foundation, the network is built on top.

---

## Why a `Module` base class

Every layer in a neural network has the same two needs:

- **Enumerate its parameters.** The training loop must be able to ask *"what are all the trainable `Value`s inside you?"*, no matter how deep the nesting goes.
- **Reset their gradients.** Before each backward pass, every parameter must be set back to zero.

Without a base class, each of `Neuron`, `Layer`, and `MLP` would have to reimplement both. With three classes, that is three copies of the same loop, three places to introduce a bug.

`Module` declares the contract once:

```python
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []
```

`zero_grad` is fully implemented in terms of `parameters()`. Subclasses only need to provide `parameters()`, and they inherit `zero_grad` for free. That is the entire reason `Module` exists: to make the "collect parameters" problem the only thing a subclass has to solve.

It is also where future methods will live — `save()`, `load()`, `train()`, `eval()`. Every one of them will follow the same pattern: defined once on `Module`, available to everything below.

---

## Why `Neuron` computes `w · x + b`

The arithmetic is the easy part. The subtleties are:

### 1. Why `sum(..., self.b)` instead of `sum(...) + self.b`

```python
act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
```

Python's `sum(iterable, start)` uses `start` as the initial accumulator value. Passing `self.b` there means the sum begins at the bias and immediately starts adding `wᵢ · xᵢ` terms on top. The bias becomes part of the graph from the very first addition.

The alternative — `sum(...) + self.b` — computes the same number, but produces a graph with one extra `+` node. In a network with thousands of neurons, that is thousands of extra nodes that serve no purpose. The first form is the canonical one, and it is why every tutorial writes it that way.

### 2. Why random weights in `[-1, 1]`

All weights initialized to zero would make every neuron in a layer compute the exact same thing. The gradient of the loss with respect to each of them would be identical, so they would all update identically, forever. The layer would collapse to a single effective neuron, no matter how many you declared.

Random initialization **breaks the symmetry**. Each neuron starts at a different point in weight space, receives a different gradient, and learns a different feature. The exact distribution matters less than the fact that it is not constant. `uniform(-1, 1)` is simple and works for small networks. Xavier, He, and others are more careful about variance, and would be the natural upgrade if you scaled this up.

### 3. Why `nonlin` is a constructor flag

The same `w · x + b` computation is used both for hidden layers (which need a nonlinearity to be useful) and for the output layer (which usually does not, because its job is to predict a raw value).

Instead of writing two classes, `Neuron` takes a flag. `nonlin=True` applies `tanh`. `nonlin=False` returns `w · x + b` unchanged. This keeps the class small and makes the distinction between "hidden neuron" and "output neuron" a matter of configuration, not of type.

That is also why `MLP` sets `nonlin=False` on the last layer: if you are doing regression and the output were passed through `tanh`, predictions would be trapped in `(-1, 1)`. The network could never predict `2.5` or `-100`, no matter how much you trained it. This is a classic bug when copying micrograd without thinking, and the flag is what prevents it.

### 4. Why `parameters()` returns `self.w + [self.b]`

The training loop only sees a flat list of `Value`s. It does not care whether each one is a weight or a bias; it just needs to reset their gradients and update their data.

`Neuron.parameters()` returns both, concatenated. The convention is: weights first, bias last. It does not matter functionally, but keeping it consistent across classes makes debugging easier when you want to inspect "the third parameter of this neuron".

---

## What comes next (and why)

`Neuron` alone is enough to hand-build a network, but the forward pass quickly becomes unreadable:

```python
layer1 = [Neuron(2) for _ in range(4)]
layer2 = [Neuron(4) for _ in range(1)]

def forward(x):
    h = [n(x) for n in layer1]
    return layer2[0](h)
```

That is fine for one hidden layer. Add three more and the `forward` function becomes a puzzle of nested lists. Worse, if you get a size wrong — `Neuron(3)` when the previous layer has 4 outputs — you will not notice until runtime, and the error message will be cryptic.

`Layer` and `MLP` solve exactly this. `Layer` groups neurons that share an input. `MLP` chains layers from a list of sizes and runs the forward pass in a loop. Adding a layer becomes editing a list of integers, and the size-matching is done once at construction time.

They are not mathematically necessary. They are the same pattern that every framework uses, because the pain they remove is real and predictable. When `nn.py` grows to include them, this document grows with them.