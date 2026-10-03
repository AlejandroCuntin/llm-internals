import numpy as np


def make_moons(n_samples=200, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out

    # Outer half-circle: points on a circle of radius 1, upper half.
    theta_out = np.linspace(0, np.pi, n_out)
    x_out = np.c_[np.cos(theta_out), np.sin(theta_out)]

    # Inner half-circle: points on a circle of radius 1, lower half,
    # shifted right and down so it interlocks with the outer one.
    theta_in = np.linspace(0, np.pi, n_in)
    x_in = np.c_[1 - np.cos(theta_in), 0.5 - np.sin(theta_in)]

    X = np.vstack([x_out, x_in])
    y = np.hstack([np.zeros(n_out), np.ones(n_in)])

    X += rng.normal(0, noise, X.shape)
    return X, y