import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from make_moons import make_moons

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import MLP
from optim import SGD


ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
)
os.makedirs(ASSETS, exist_ok=True)


def main():
    # Reproducibility
    random.seed(42)
    np.random.seed(42)

    # 1. Dataset: two interleaving half-circles

    X, y = make_moons(n_samples=200, noise=0.1, seed=42)

    # Wrap as plain Python lists of floats: our engine works on scalars.
    xs = [[float(x0), float(x1)] for x0, x1 in X]
    ys = [float(label) for label in y]

    print(f"dataset: {len(xs)} samples, 2 features, binary labels")


    # 2. Model and optimizer

    model = MLP(2, [8, 8, 1])
    optimizer = SGD(model.parameters(), lr =0.1)

    n_steps = 500
    losses = []

    # 3 Training loop.

    for step in range(n_steps):
        # Forward pass: predictions for every sample
        preds = [model(x) for x in xs]

        # MSE loss, averaged over the dataset
        loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(xs)
        losses.append(loss.data)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step:4d}   loss {loss.data:.4f}")

    print(f"final loss: {losses[-1]:.4f}")

    # 4 loss curve

    plt.figure(figsize=(7, 4))
    plt.plot(losses, color="#2c7fb8", linewidth=2)
    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    plt.title("Training loss on two moons")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(ASSETS, "moons_loss.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {loss_path}")

    # 5 Decision boundary.
    # Dense grid over the input space, classify each point
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )

    # Predict on every grid point. This is slow with scalars, so we
    # only use a coarser grid (h=0.05 instead of 0.02)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    preds_grid = [model([float(a), float(b)]).data for a, b in grid_points]
    Z = np.array(preds_grid).reshape(xx.shape)

    plt.figure(figsize=(7, 5.5))
    # Decision boundary: where prediction crosses 0.5
    plt.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

    # Scatter the dataset on top.
    colors = ["#d62728" if label == 1 else "#1f77b4" for label in ys]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors="k",
                s=40, linewidths=0.8, zorder=3)

    plt.title("Decision boundary after training")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.tight_layout()
    db_path = os.path.join(ASSETS, "moons_decision_boundary.png")
    plt.savefig(db_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {db_path}")


if __name__ == "__main__":
    main()