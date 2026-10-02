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

