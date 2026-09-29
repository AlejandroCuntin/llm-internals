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

# Layer 

class Layer(Module):
    """
    A layer of neurons, all receiving the same input vector x.
    Returns a list of outputs (or a single Value if the layer has one neuron, to make the transition
    to the next layer seamless).
    """
    
    def __init__(self,nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        #Single-neuron layer returns a bare Value, not a list of one, #so downstream code does not need to
        # unwrap it
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def _repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

# MLP

class MLP(Module):
    """
    A multi-layer perceptron: a stack of layers.

    Constructed from a list of layer sizes. The last layer is always linear (no tanh), because its output
    is a prediction, not an activation. Applying tanh there would clamp predictions to (-1,1)
    and make regression impossible.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin = (i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"