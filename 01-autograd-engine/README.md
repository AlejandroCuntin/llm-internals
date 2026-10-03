# Autograd Engine — Project Overview

Minimal autograd engine + neural network library built from scratch (inspired by micrograd).

---

## Core Modules

| File | Purpose |
|------|---------|
| **engine.py** | Core `Value` class: scalar autograd with topological-order reverse-mode backprop. Implements `+`, `*`, `/`, `**`, `tanh`, `exp`, `log`, `relu`, `sigmoid`. |
| **nn.py** | Neural net building blocks: `Module` (base), `Neuron` (w·x+b + tanh), `Layer` (list of neurons), `MLP` (stack of layers; last layer linear). |
| **optim.py** | `SGD` optimizer: `step()` updates params via `p.data -= lr * p.grad`. |
| **viz.py** | GraphViz visualization: `trace()` walks the computation graph, `draw_dot()` renders nodes (data/grad) and op boxes as SVG. |
| **nn_viz.py** | Matplotlib network diagram: draws MLP layers as circles, connections colored by weight sign/width by magnitude, neuron fill by activation value. |

---

## Examples (in `examples/`)

| Script | What it does | Output |
|--------|--------------|--------|
| `visualize_graph.py` | Builds `f = (a*b + a.tanh())*2`, renders graph before/after `backward()` | `assets/graph_before_backward.svg`, `graph_after_backward.svg` |
| `walkthrough_visual.py` | Step-by-step graph construction (Value → inputs → mul → add → backward), generates HTML walkthrough | `assets/Walkthrough_*.svg`, `walkthrough.html` |
| `draw_network.py` | Draws MLP(2→4→4→1) architecture clean + with forward-pass activations | `assets/network_clean.svg`, `network_with_values.svg` |
| `train_moons.py` | Trains MLP(2→8→8→1) on two-moons dataset (500 steps, SGD), plots loss curve + decision boundary | `assets/moons_loss.png`, `moons_decision_boundary.png` |
| `animate_training.py` | Animates MLP(2→3→1) training on sin·cos data, saves GIF of network evolving | `assets/training_animation.gif` |
| `gradient_evolution.py` | Trains MLP(2→8→8→1), tracks per-layer gradient L2 norms + loss (log scale) | `assets/gradient_evolution.png` |
| `make_moons.py` | Dataset generator: two interleaving half-circles with noise | — |

---

## Assets (in `assets/`)

| File | Description |
|------|-------------|
| `graph_before_backward.svg` | Computation graph before gradients; nodes show `data`, `grad=0` |
| `graph_after_backward.svg` | Same graph after `backward()`; `grad` fields populated |
| `Walkthrough_ 01_Value.svg` … `Walkthrough_ 05_backward.svg` | 5-step visual progression: single node → two inputs → multiply → add (shared input) → backward |
| `walkthrough.html` | Browser-friendly page stitching the 5 SVGs with explanations |
| `network_clean.svg` | MLP architecture diagram (no activations) |
| `network_with_values.svg` | Same network with forward-pass activations shown per neuron |
| `moons_loss.png` | Training loss curve (MSE vs step) for two-moons classification |
| `moons_decision_boundary.png` | Decision boundary contour (black line at 0.5) overlaid on dataset scatter |
| `training_animation.gif` | 60-frame GIF: network weights/activations evolving during training |
| `gradient_evolution.png` | Two panels: (left) log-scale loss decay, (right) log-scale gradient L2 norm per layer over training |