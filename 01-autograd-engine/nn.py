import random

from engine import Value

# Base Class

class Module:
    """
    Base class for anything with trainable parameters.

    Subclasses must implement parameters(). They inherit zero_grad(), which walks that list
    and resets each parameter's gradient to 0.
    """

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return[]

# Neuron

class Neuron(Module):
    """
    A single neuron: computes w * x + b, then applies tanh if nonlin=True.
    """

    def __init__(self, nin, nonlin=True):
        #Small random weights in [-1, 1] to breack symmetry. All zeros
        #would make every neuron in a layer learn the same thing.

        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        #List of Value weights, one per input

        self.b = Value(0.0)
        #Value bias

        self.nonlin = nonlin    

    def __call__(self, x):
        #sum(...., self.b) uses b as the starting value of the sum.,
        #so the bias is part of the graph from the start.

        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        kind = "Tanh" if self.nonlin else "Linear"
        return f"{kind}Neuron({len(self.w)})"