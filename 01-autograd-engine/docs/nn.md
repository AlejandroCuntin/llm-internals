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

---

## Why `Layer` exists

`Layer` is the first class that is not strictly necessary. You could write a network with only `Neuron` objects in a flat list, and it would work. So why add it?

The problem is **grouping**. A layer is a set of neurons that all receive the same input and produce independent outputs. Without a class, you represent this with a Python list:

```python
hidden = [Neuron(2), Neuron(2), Neuron(2), Neuron(2)]
```

That works until you want to compute the forward pass:

```python
def forward(x):
    return [n(x) for n in hidden]
```

Now multiply that by three layers and you have the same list comprehension three times. If you ever want to inspect the layer — count its neurons, print its weights, debug why one of them is not learning — you are operating on a bare list with no identity.

`Layer` gives the group a name. It is still a list of neurons internally, but the list is now a *thing* you can pass around, call, and ask for parameters.

### 1. Why `Layer.__call__` returns a single Value sometimes

```python
def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
```

The last layer of a regression network usually has exactly one neuron: the one that produces the prediction. If `Layer` always returned a list, the output of the network would be `[Value]`, a list of one element, and every piece of downstream code — loss computation, printing, plotting — would have to unwrap it.

By returning a bare `Value` when the layer has one neuron, and a list otherwise, the interface stays natural in both cases. `model(x)` returns a `Value` for regression, a list for multi-class classification, and neither case requires the caller to remember which one it is.

Is this "clean"? It is a small inconsistency. Is it worth it? For this project, yes. The alternative — always returning a list — would spread `[0]` unwrapping across every example and every loss function. In a larger framework, the type would be explicit (`Tensor` with a shape), and the ambiguity would disappear. Here, the convenience wins.

### 2. Why `parameters()` flattens

```python
def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]
```

The training loop does not want a nested structure. It wants one flat list of `Value`s that it can iterate over once per step. `Layer.parameters()` is the adapter that turns "a list of neurons, each with its own parameters" into "a list of parameters".

Every class in `nn.py` follows the same convention: `parameters()` always returns a flat list. `MLP` calls `Layer.parameters()` for each layer and concatenates. `Layer` calls `Neuron.parameters()` for each neuron and concatenates. `Neuron` returns its own weights and bias. The result is that `mlp.parameters()` gives you everything, from the top level, without any nesting.

This is what makes `model.zero_grad()` work with one line. If `parameters()` returned nested structures, `zero_grad` would need a recursive walker.

---

## Why `MLP` exists

With `Neuron` and `Layer`, you can already build a network:

```python
l1 = Layer(2, 4)
l2 = Layer(4, 4)
l3 = Layer(4, 1)

def forward(x):
    x = l1(x)
    x = l2(x)
    x = l3(x)
    return x
```

That is readable. The problem appears the moment you change the architecture: adding a layer means declaring a new `Layer`, adding a new line to `forward`, and verifying by hand that the output size of one layer matches the input size of the next. Miss one number and the error only shows up at runtime, in a shape mismatch deep inside `zip()`.

`MLP` removes both problems by making the architecture a **list of integers**:

```python
model = MLP(2, [4, 4, 1])
```

Everything else — building the layers, chaining them, running the forward pass — is done by the class. Adding a hidden layer is editing the list. Size matching is performed once at construction time, so a mismatched architecture fails immediately, not three function calls deep.

### 1. Why the last layer is linear

```python
nonlin=(i != len(nouts) - 1)
```

This single expression is the most important line in `MLP`. It says: *"every layer applies tanh, except the last one, which is linear."*

The reason is that `tanh` clamps its output to `(-1, 1)`. For hidden layers, that is exactly what you want: bounded activations keep the signal from exploding as it propagates through many layers. For the output layer of a regression model, it is catastrophic. If the network is trying to predict a house price of `450000`, it can never reach it: `tanh` would cap the prediction at `1.0`, and no amount of training would fix it. The loss would plateau, the gradients would vanish, and you would spend hours looking for a bug that is not there.

The fix is one boolean. The bug it prevents is one of the most common when people copy micrograd without understanding it.

For classification, the situation is different: you usually want a `sigmoid` (binary) or `softmax` (multi-class) on the output. But those are applied *outside* the MLP, on top of a linear score. That keeps `MLP` agnostic about the task and lets the same class serve both regression and classification by choosing what to wrap around its output.

### 2. Why sizes are computed at construction time

```python
sz = [nin] + nouts
self.layers = [
    Layer(sz[i], sz[i + 1], ...)
    for i in range(len(nouts))
]
```

`sz` is the full list of layer sizes, from input to output: `[nin, n1, n2, ..., nout]`. Each `Layer` is built from a consecutive pair `(sz[i], sz[i+1])`, so the output of one layer is guaranteed to be the input size of the next. There is no way to mismatch them.

If you tried to build the layers manually — `Layer(2, 4)`, `Layer(4, 4)`, `Layer(4, 1)` — you could mistype a number and only find out at runtime. `MLP` makes that class of error impossible. The architecture is a single source of truth: the list `nouts`. Everything else is derived from it.

### 3. Why the forward pass is a loop

```python
def __call__(self, x):
    for layer in self.layers:
        x = layer(x)
    return x
```

This is the same pattern as `parameters()`: an operation that is naturally recursive, expressed as an iteration over a flat collection. The forward pass of any MLP, regardless of depth, is "apply each layer in order". A `for` loop says exactly that. There is no special case for the first layer or the last one; the single-neuron unwrapping in `Layer.__call__` handles the only asymmetry, and it is handled at the right level.

This is also why `MLP` composes with itself. You could wrap an MLP in another module that treats it as a single layer, and the same forward loop would still work. The pattern scales.

---

## The design in one paragraph

`Module` declares the contract: *"I know how to enumerate my parameters, and therefore how to reset their gradients."* `Neuron` is the atomic unit: a weighted sum plus a bias, with an optional nonlinearity. `Layer` groups neurons that share an input and flattens their parameters into one list. `MLP` chains layers from a list of sizes and runs the forward pass as a loop, ensuring that size matching happens once, at construction, and that the last layer is linear so the network can predict values outside `(-1, 1)`.

Each class exists because the previous one, on its own, made some part of building or training a network painful. The pain is real and predictable, and the class is the smallest thing that removes it. That is the only test a class needs to pass.