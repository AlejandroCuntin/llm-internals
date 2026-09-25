import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        # data: scalar value this node holds
        # grad: gradient of final output w.r.t this node (accumulated during backward)
        # _prev: tuple of child nodes in the computation graph
        # _op: string label of the operation that produced this node (for debugging)
        # _backward: closure that computes local gradients and accumulates into children's grad
        self.data = data
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        # ensure other is a Value node (supports scalar + Value)
        other = other if isinstance(other, Value) else Value(other)
        # forward pass: create output node with children (self, other) and op '+'
        out = Value(self.data + other.data, (self, other), '+')

        # backward pass: d/dx (x + y) = 1, so gradient flows unchanged to both inputs
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        # d/dx (x * y) = y, d/dy (x * y) = x  (product rule)
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self):         return self * -1
    def __sub__(self, other):  return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other):  return self * (other ** -1)
    def __rtruediv__(self, other): return other * (self ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only numeric exponents"

        # negative base with non-integer exponent produces complex numbers
        if self.data < 0 and not float(other).is_integer():
            raise ValueError(
                f"Negative base with non-integer exponent: {self.data} ** {other}"
            )

        out = Value(self.data ** other, (self,), f"**{other}")

        # d/dx (x^n) = n * x^(n-1)
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        # d/dx tanh(x) = 1 - tanh^2(x)
        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        # exp(709) ~ 8.2e307, near float64 max (~1.8e308)
        if self.data > 709.0:
            raise OverflowError(f"exp({self.data}) overflows float64")

        out = Value(math.exp(self.data), (self,), 'exp')

        # d/dx exp(x) = exp(x) = out.data
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        assert self.data > 0, f"log requires x > 0, received {self.data}"
        out = Value(math.log(self.data), (self,), 'log')

        # d/dx log(x) = 1/x
        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'relu')

        # d/dx relu(x) = 1 if x > 0 else 0
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        # d/dx sigmoid(x) = s * (1 - s)
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def __float__(self): return float(self.data)
    def __int__(self):   return int(self.data)

    def __abs__(self):
        return self if self.data >= 0 else -self

    def __lt__(self, other): return self.data <  (other.data if isinstance(other, Value) else other)
    def __le__(self, other): return self.data <= (other.data if isinstance(other, Value) else other)
    def __gt__(self, other): return self.data >  (other.data if isinstance(other, Value) else other)
    def __ge__(self, other): return self.data >= (other.data if isinstance(other, Value) else other)

    def __eq__(self, other):
        # identity check for Value objects, value equality for scalars
        if isinstance(other, Value):
            return self is other
        return self.data == other

    __hash__ = object.__hash__

    def _topo(self):
        # iterative DFS topological sort (post-order)
        # returns nodes in order such that children appear before parents
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))          # mark for post-order append
            for child in node._prev:
                stack.append((child, False))
        return topo

    def zero_grad(self):
        # reset gradients for all nodes in the graph
        for node in self._topo():
            node.grad = 0.0

    def backward(self):
        # seed gradient of output node as 1.0 (dL/dL = 1)
        self.grad = 1.0
        # traverse in reverse topological order (children before parents)
        # so each node's _backward sees its out.grad already accumulated
        for node in reversed(self._topo()):
            node._backward()