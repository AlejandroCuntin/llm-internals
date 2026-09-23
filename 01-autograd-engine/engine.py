import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

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

        if self.data < 0 and not float(other).is_integer():
            raise ValueError(
                f"Negative base with non-integer exponent: {self.data} ** {other}"
            )

        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        if self.data > 709.0:
            raise OverflowError(f"exp({self.data}) overflows float64")

        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        assert self.data > 0, f"log requires x > 0, received {self.data}"
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

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
        if isinstance(other, Value):
            return self is other
        return self.data == other

    __hash__ = object.__hash__

    def _topo(self):
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
            stack.append((node, True))
            for child in node._prev:
                stack.append((child, False))
        return topo

    def zero_grad(self):
        for node in self._topo():
            node.grad = 0.0

    def backward(self):
        self.grad = 1.0
        for node in reversed(self._topo()):
            node._backward()