import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import MLP
from optim import Adam

N_SAMPLES = 200       # how many images to train on
EPOCHS = 3            # how many passes over the data
LR = 0.01
HIDDEN = [16]         # one hidden layer of 16 neurons

def load_digits():
    """Load the 8x8 digits dataset. Returns (X, y) as numpy arrays."""
    from sklearn.datasets import load_digits as _load
    data = _load()
    return data.images, data.target

def preprocess(X, y, n):
    """
    Take n samples, flatten images to 64, normalize to [-1, 1],
    one-hot encode labels.
    """
    X = X[:n].reshape(n, -1).astype(np.float64)
    y = y[:n]
    X = (X - 8.0) / 8.0                # pixels 0..16 -> -1..1
    Y = np.eye(10)[y]                  # one-hot, shape (n, 10)
    return X, y, Y


def train_one_sample(model, opt, x, target):
    """
    One stochastic gradient step on a single sample.

    Returns the loss value and the list of 10 output Values.
    """
    preds = model(x)                                       # list of 10 Values
    loss = sum((p - t) ** 2 for p, t in zip(preds, target)) / 10

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.data, preds


def main():
    X, y, Y = preprocess(*load_digits(), N_SAMPLES)

    model = MLP(64, HIDDEN + [10])
    opt = Adam(model.parameters(), lr=LR)

    n_params = len(model.parameters())

    print()
    print(f"  Dataset:    {N_SAMPLES} images (8x8 = 64 pixels, 10 classes)")
    print(f"  Network:    64 -> {' -> '.join(map(str, HIDDEN))} -> 10")
    print(f"  Parameters: {n_params}")
    print(f"  Optimizer:  Adam (lr={LR})")
    print()
    print(f"  Running {EPOCHS} epochs of stochastic gradient descent.")
    print(f"  One sample at a time. Watch the per-epoch timing.")
    print()

    for epoch in range(EPOCHS):
        t0 = time.time()
        epoch_loss = 0.0
        correct = 0

        for i in range(N_SAMPLES):
            x = X[i].tolist()
            target = Y[i].tolist()

            loss_val, preds = train_one_sample(model, opt, x, target)
            epoch_loss += loss_val

            # argmax of the 10 outputs
            pred_class = int(np.argmax([p.data for p in preds]))
            if pred_class == y[i]:
                correct += 1

            # Live progress every 50 samples so we know it isn't hung.
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                ms = elapsed / (i + 1) * 1000
                print(f"    [{i+1:>4}/{N_SAMPLES}]  "
                      f"avg {ms:.0f} ms/sample  "
                      f"elapsed {elapsed:.1f}s")

        elapsed = time.time() - t0
        avg_loss = epoch_loss / N_SAMPLES
        acc = correct / N_SAMPLES
        ms_per_sample = elapsed / N_SAMPLES * 1000

        print()
        print(f"  epoch {epoch+1}/{EPOCHS}  "
              f"loss {avg_loss:.4f}  "
              f"acc {correct}/{N_SAMPLES} ({acc*100:.1f}%)  "
              f"time {elapsed:.1f}s  "
              f"({ms_per_sample:.0f} ms/sample)")
        print()

    print("  Done.")
    print()
    print("  If a single epoch took more than a few seconds, you have just")
    print("  felt the ceiling of the scalar engine. That is the point of")
    print("  this script: the maths is correct, the speed is not.")
    print()


if __name__ == "__main__":
    main()