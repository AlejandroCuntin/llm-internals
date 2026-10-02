import math
import os
import random
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import MLP


def target(x):
    return math.sin(x[0]) * math.cos(x[1])


def make_dataset(n=32):
    random.seed(42)
    xs, ys = [], []
    for _ in range(n):
        x = [random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)]
        xs.append(x)
        ys.append(target(x))
    return xs, ys

def activation_color(value):
    v = max(-1.0, min(1.0, value))
    if v >= 0:
        return (1.0, 1.0 - v, 1.0 - v)
    return (1.0 + v, 1.0 + v, 1.0)


def draw_state(ax, model, x_input):
    """Clear the axis and draw the current state of the network."""
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")

    layers = model.layers
    sizes = [len(layers[0].neurons[0].w)]
    for layer in layers:
        sizes.append(len(layer.neurons))
    n_layers = len(sizes)
    layer_spacing = 2.0

    #Compute activations

    activations = [list(x_input)]
    h = x_input
    for layer in layers:
        h = layer(h)
        if not isinstance(h, list):
            h = [h]
        activations.append([v.data for v in h])

    #Position
    positions = []
    for i, size in enumerate(sizes):
        x_pos = i * layer_spacing
        y_pos = [j - (size - 1) / 2 for j in range(size)]
        positions.append([(x_pos, y) for y in y_pos])

    #Connections

    for i, layer in enumerate(layers):
        for j, neuron in enumerate(layer.neurons):
            x1, y1 = positions[i + 1][j]
            for k, w in enumerate(neuron.w):
                x0, y0 = positions[i][k]
                color = "#d62728" if w.data < 0 else "#2ca02c"
                width = min(abs(w.data) * 2.5, 4.5) + 0.3
                alpha = min(abs(w.data) + 0.3, 1.0)
                arrow = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0.08",
                    color=color,
                    linewidth=width,
                    alpha=alpha,
                    zorder=1,
                )
                ax.add_patch(arrow)

    #Neurons
    for i, layer_positions in enumerate(positions):
        for j, (x0, y0) in enumerate(layer_positions):
            if i == 0:
                color = "white"
            else:
                color = activation_color(activations[i][j])
            ax.add_patch(Circle(
                (x0, y0), radius=0.32,
                facecolor=color, edgecolor="#333",
                linewidth=1.5, zorder=3,
            ))

    # Layer labels
    labels = ["input"] + [f"hidden {i}" for i in range(1, n_layers - 1)] + ["output"]
    for i, label in enumerate(labels):
        ax.text(i * layer_spacing, max(sizes) / 2 + 1.0, label,
                ha="center", va="bottom", fontsize=10, color="#555")

    ax.set_xlim(-0.8, (n_layers - 1) * layer_spacing + 0.8)
    ax.set_ylim(-max(sizes) / 2 - 1.2, max(sizes) / 2 + 1.8)

def main():
    ASSETS = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets",
    )
    os.makedirs(ASSETS, exist_ok=True)

    # Small model, small dataset, few steps.
    random.seed(0)
    xs, ys = make_dataset()
    model = MLP(2, [3, 1])
    x_sample = [0.5, -0.3]
    lr = 0.05
    n_frames = 60
    steps_per_frame = 3

    fig, ax = plt.subplots(figsize=(7, 5))

    def update(frame):
        # Run a few training steps between frames.
        for _ in range(steps_per_frame):
            preds = [model(x) for x in xs]
            loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(xs)
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= lr * p.grad

        draw_state(ax, model, x_sample)
        ax.set_title(f"frame {frame} — loss = {loss.data:.4f}", fontsize=12)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=150, repeat=False)

    out = os.path.join(ASSETS, "training_animation.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    print(f"Saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()