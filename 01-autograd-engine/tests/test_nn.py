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

# Layer

def test_layer_single_neuron_returns_value():
    l = Layer(2,1)
    out = l([1.0,2.0])
    assert isinstance(out, Value)

def test_layer_mutiple_neurons_returns_list():
    l = Layer(2,4)
    out = l([1.0, 2.0])
    assert isinstance(out,list)
    assert len(out) == 4

def test_layer_parameter_count():
# 5 neurons * (3 weights + 1 bias) = 20
    assert Layer(3,5).parameters() == 20
    

# MLP - Structure

def test_mlp_parameters_count():
    # [2, 4, 4, 1] -> 4*3 + 4*5 + 1*5 = 12 + 20 + 5 = 37
    assert len(MLP(2, [4,4,1]).parameters()) == 37

def test_mlp_last_layer_is_linear():
    # If this fails, someone removed the nonlin=False on the last layer.
    m = MLP(2, [4,1])
    assert m.layers[-1].neurons[0].nonlin is False

def test_mlp_hidden_layers_are_tanh():
    m = MLP(2, [4,4,1])
    assert m.layers[0].neurons[0].nonlin is True
    assert m.layers[1].neurons[0].nonlin is True

# MLP - forward and backward

def test_mlp_forward_shaper():
    m = MLP(2, [4,1])
    out = m([1.0, 2.0])
    assert isinstance(out,Value)

def test_mlp_backward_populates_grads():
    m = MLP(2, [4,4,1])
    out = m([1.0, 2.0])
    out.backward()
    #at least some parameter must receive a nonzero gradient, otherwise the backward pass did not
    #reach the leaves.
    grads = [p.grad for p in m.parameters()]
    assert any(g != 0.0 for g in grads)

#Runner

if __name__ == "__main__":
    tests = [(name, obj) for name, obj in sorted(globals().items())
            if name.startwith("test_") and callable(obj)]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL {name}: {e}")  
            failed += 1
    print(f"\n {passed} passed, {failed} failed")