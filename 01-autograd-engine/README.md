# Phase 1: Autograd Engine

A autograd engine that builds computational graphs and calculates gradients via backpropagation. This is the mathematical heart of the
neural network we'll build in later phases.

## What's inside

| File | Purpose |
|---|---|
| `engine.py` | The `Value` class: a number that remembers how it was created |
| `viz.py` | Visualize the computational graph with Graphviz |
| `nn.py` | `Neuron`, `Layer`, `MLP` built on top of `Value` |
| `optim.py` | Optimizers (SGD, Momentum, Adam) |

