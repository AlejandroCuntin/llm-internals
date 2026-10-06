class SGD:
    """
    Stochastic Gradient Descent.
    params: list of Value objects to update
    lr: learning rate
    """
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


class Momentum:
    """
    SGD with momentum.
    Maintains a velocity per parameter that accumulates past gradients.
    This smooths the trajectory and helps escape shallow local minima.
    """

    def __init__(self, params, lr=0.01, beta=0.9, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = [0.0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            grad = p.grad + self.weight_decay * p.data
            self.velocities[i] = self.beta * self.velocities[i] + grad
            p.data -= self.lr * self.velocities[i]

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


class Adam:
    """
    Adam optimizer.
    Maintains a running estimate of the first moment (mean) and second
    moment (uncentered variance) of the gradients, and uses them to
    scale the update per parameter.
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [0.0 for _ in self.params]
        self.v = [0.0 for _ in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            grad = p.grad + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0