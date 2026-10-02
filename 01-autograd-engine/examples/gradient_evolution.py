import math
import os
import random
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import MLP

def target(x):
    return math.sin(x[0]) * math.cos(x[1])


def make_dataset(n=64):
    random.seed(42)
    xs = []
    ys = []
    for _ in range(n):
        x = [random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0)]
        xs.append(x)
        ys.append(target(x))
    return xs, ys

def grad_norm(layer):
    """L2 norm of all gradients in a layer."""
    total = 0.0
    for p in layer.parameters():
        total += p.grad ** 2
    return math.sqrt(total)


# Training loop with gradient tracking

def train(steps= 200, lr=0.05, record_every=1):
    random.seed(0)

    xs, ys = make_dataset()
    model = MLP(2, [8,8,1])
    losses = []
    grad_norms_per_layer = [[] for _ in model.layers]
    steps_recorded = []

    for step in range(steps):
        #Forward: compute predictions and MSE loss.
        preds = [model(x) for x in xs]
        loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(xs)

        # Backward
        model.zero_grad()
        loss.backward()

        # Record before updating (this is the gradient the optimizer sees).
        if step % record_every == 0:
            losses.append(loss.data)
            steps_recorded.append(step)
            for i, layer in enumerate(model.layers):
                grad_norms_per_layer[i].append(grad_norm(layer))

        # SGD step

        for p in model.parameters():
            p.data -= lr * p.grad

    return model, losses, steps_recorded, grad_norms_per_layer



def plot_training(steps, losses, grad_norms, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: loss curve
    ax1.plot(steps, losses, color="#2c7fb8", linewidth=2)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("loss (MSE)")
    ax1.set_title("Loss over training")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")   # log scale makes the decay visible

    # Right: gradient norm per layer
    colors = ["#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e"]
    for i, norms in enumerate(grad_norms):
        ax2.plot(steps, norms, label=f"layer {i}",
                 color=colors[i % len(colors)], linewidth=1.8)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("gradient L2 norm")
    ax2.set_title("Gradient magnitude per layer")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved {save_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    ASSETS = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets",
    )
    os.makedirs(ASSETS, exist_ok=True)

    model, losses, steps, grad_norms = train(steps=300, lr=0.05)

    print(f"final loss: {losses[-1]:.6f}")
    print(f"initial loss: {losses[0]:.6f}")

    plot_training(
        steps, losses, grad_norms,
        save_path=os.path.join(ASSETS, "gradient_evolution.png"),
)