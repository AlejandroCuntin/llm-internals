import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import MLP
from optim import SGD, Momentum, Adam
from make_moons import make_moons


ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
)
os.makedirs(ASSETS, exist_ok=True)


def train(model, optimizer, xs, ys, n_steps):
    losses = []
    for step in range(n_steps):
        preds = [model(x) for x in xs]
        loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(xs)
        losses.append(loss.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 25 == 0:
            print(f"    step {step:4d}  loss {loss.data:.4f}")

    return losses


def main():
    random.seed(42)
    np.random.seed(42)

    X, y = make_moons(n_samples=60, noise=0.1, seed=42)
    xs = [[float(a), float(b)] for a, b in X]
    ys = [float(label) for label in y]

    n_steps = 150

    results = {}
    for name, lr in [("SGD", 0.1), ("Momentum", 0.05), ("Adam", 0.001)]:
        random.seed(0)
        model = MLP(2, [6, 6, 1])  

        if name == "SGD":
            opt = SGD(model.parameters(), lr=lr)
        elif name == "Momentum":
            opt = Momentum(model.parameters(), lr=lr, beta=0.9)
        else:
            opt = Adam(model.parameters(), lr=lr)

        losses = train(model, opt, xs, ys, n_steps)
        results[name] = losses
        print(f"{name:>10}  final loss: {losses[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    colors = {"SGD": "#1f77b4", "Momentum": "#ff7f0e", "Adam": "#2ca02c"}
    for name, losses in results.items():
        plt.plot(losses, label=name, color=colors[name], linewidth=2)

    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    plt.title("Optimizer comparison on two moons")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(ASSETS, "optimizer_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()