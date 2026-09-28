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

