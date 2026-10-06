import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import MLP
from optim import Adam
from make_moons import make_moons

N_SAMPLES = 30
N_STEPS = 150
HIDDEN = [6, 6]

# The values to sweep. Edit this list to try different learning rates.
LR_VALUES = [0.001, 0.003, 0.010, 0.030]


# Training

def train_one(lr, xs, ys, n_steps=N_STEPS):
    """Train a fresh MLP with Adam at the given lr. Return the final loss."""
    # Reset the seed before each run so the initialization is identical.
    # Only the lr changes between runs.
    random.seed(0)
    model = MLP(2, HIDDEN + [1])
    opt = Adam(model.parameters(), lr=lr)

    final_loss = None
    for _ in range(n_steps):
        preds = [model(x) for x in xs]
        loss = sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(xs)
        final_loss = loss.data

        opt.zero_grad()
        loss.backward()
        opt.step()

    return final_loss

def main():
    
    random.seed(42)
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.1, seed=42)
    xs = [[float(a), float(b)] for a, b in X]
    ys = [float(label) for label in y]

    print(f"Adam learning rate sweep")
    print(f"  samples:  {N_SAMPLES}")
    print(f"  steps:    {N_STEPS}")
    print(f"  network:  2 -> {' -> '.join(map(str, HIDDEN))} -> 1")
    print()

    results = []
    for lr in LR_VALUES:
        final = train_one(lr, xs, ys)
        results.append((lr, final))
        print(f"  Adam lr={lr:<6}  final loss: {final:.4f}")

    # Highlight the best.
    best_lr, best_loss = min(results, key=lambda r: r[1])
    print()
    print(f"  best: lr={best_lr}  loss={best_loss:.4f}")


if __name__ == "__main__":
    main()