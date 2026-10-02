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



