class Value:
    """
    Stores a single number and its gradient. It also remembers how it was created to calculate derivates
    later.
    """

    def __unit__(self, data, _children=(), _op=""):
        self.data = data
        #Gradient starts at 0. It represents how much this value affects the end result
        self.grad = 0.0
        #The 'parents" that created this value. We use a saet so we don't repeat nodes
        self._prev = set(_children)
        self._op = _op
        #A function to calculate the gradient. By default, it does nothing
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad = {self.grad})"

    def __add__(self, other):
        #If other is a normal number, we convert it into a Value object
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self,other), '+')

        def _backward():
            #We use += instead of = because a value might be used in multiple sums
            #The derivative of addition is 1, multiplied by the incoming gradient.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
    #If "other" is a regular number, like 3, we wrap iit in a Value Object
    #so operatoins like 'a * 3' work smoothly without crashing
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self,other), '*')

    def _backward():

        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
        
        