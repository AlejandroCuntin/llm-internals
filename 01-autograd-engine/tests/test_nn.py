import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Value
from nn import *

#Helper
def assert_raises(exc,fn):
    try:
            fn()
    except exc:
        return
    except Exception as other:
        raise AssertionError(
            f"expected { exc.__name__}, got {type(other).__name__}: {other}"
        )
    raise AssertionError(f"expceted {exc.__name__}, nothing was raised")



#Neuron, dimension check

def test_neuron_dimension_mismatch_raises():
    #If Neuron does not assert len(x) == len(self.w), zip() silently truncates and the extra input is ignored
    # This locks the fix in.
    n = Neuron(3)
    assert_raises(AssertionError, lambda: n([1.0 , 2.0]))

def test_neuron_accepts_correct_dimension():
    n = Neuron(3)
    out = n([1.0,2.0,3.0])
    assert isinstance(out,Value)

#Neuron forward pass


def test_neuron_forward_uses_all_inputs():
    #With zip truncation, changing the last input would not affect the output. This test would then fail
    n = Neuron(3, nonlin = False)

    n.w = [Value(1.0), Value(1.0), Value(1.0)]
    n.b = Value(0.0)
    y1 = n([1.0,1.0,1.0]).data
    y2 = n([1.0,1.0,1.0]).data
    assert y2 != y1 #third input matters

def test_neuron_tanh_clips_output():
    n = Neuron(2, nonlin=True)
    n.w = [Value(10.0), Value(10.0)]
    n.b = Value(0.0)
    assert abs(n([1.0, 1.0]).data) < 1.0

def test_neuron_linear_no_tanh():
    #nonlin = False sould return the raw w * x + b, not clipped
    n = Neuron(2, nonlin=False)
    n.w = [Value(10.0), Value(10.0)]
    n.b = Value(0.0)
    assert n([1.0, 1.0]).data == 20.0   

#Neuron / Module, parameters and zero_grad

def test_neuron_parameter_count():
    n = Neuron(3)
    assert len(n.parameters()) == 4 # 3 weights + 1 bias = 4

def test_neuron_zero_grad_resets():
    n = Neuron(2)
    for p in n.parameters():
        p.grad = 5.0
    n.zero_grad()
    for p in n.parameters():
        assert p.grad == 0.0