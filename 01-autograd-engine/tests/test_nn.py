import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Value
from nn import *

#Helper
def assert_raises(exc,fn):
    

#Neuron
def test_neuron_dimension_mismatch_raises():
    n = Neuron(3)
    assert_raises(AssertionError, lambda: n([1.0 , 2.0]))

def test_neuron_parameter_count():
    n = Neuron(3)
    assert len(n.parameters()) == 4 # 3 weights + 1 bias

def test_neuron_forward_uses_all_inputs():
    #If zip truncated silently, chaging the last input would not 
    # change the output. This catches the bug
    n = Neuron(3, nonlin = False)
    #fix weights

    n.w = [Value(1.0), Value(1.0), Value(1.0)]
    n.b = Value(0.0)
    y1 = n([1.0,1.0,1.0]).data
    y2 = n([1.0,1.0,1.0]).data
    assert y2 != y1 #third input matters

def test_neuron_nonlin_flag():
    n = Neuron(2, nonlin=False)
    n.w = [Value(10.0), Value(10.0)]
    n.b = Value(0.0)
    # linear: output should be 20 (not clipped to 1)
    assert n([1.0, 1.0]).data == 20.0

    m = Neuron(2, nonlin=True)
    m.w = [Value(10.0), Value(10.0)]
    m.b = Value(0.0)
    # tanh: output is clipped
    assert m([1.0, 1.0]).data < 1.0