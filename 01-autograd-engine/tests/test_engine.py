import typing_extensions
import os
import sys

#Try the normal import first: works when the test is run from the project root, where engine.py lives.
try:
    from engine import Value
except ImportError:
    #Fallback: if the import fails (e.g. running from inside tests/),
    #add the parent directory to sys.path so Python can find engine.py
    sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine import Value

def numeric_grad(f,datas,idx, eps=1e-6):
    """
    Central-difference derivative of f w.r.t. datas[idx].
    f is called with fresh Value objects, so the original graph is never touched.
    """

    plus = [Value(d) for d in datas]
    plus[idx].data +=eps
    f_plus = f(*plus).data 

    #Same, but with the input nudged in the opposite direction
    minus = [Value(d) for d in datas]
    minus[idx].data -= eps
    f_minus = f(*minus).data

    #Central difference formula: (f(x+h) - f(x-h)) / 2h
    #More accurate than forward difference (f(x + h ) - f(x)) / h.

    return (f_plus - f_minus) / (2* eps)


def check (f, *datas, tol= 1e-5):
    """
    Verify that backward)() matches numeric gradients for each input of f.
    """

    #Wrap each raw number in a Value so we can call backward().

    values =  [Value(d) for d in datas]
    out = f(*values) #build the graph 
    out.backward() #fill .grad on every node

    #Compare the analytic gradient (from backward) with the numeric one (from finite differences) for
    #each input.numeric_grad
    for i in range(len(datas)):
        analytic = values[i].grad
        numeric = numeric_grad(f,datas,i)
        assert abs(analytic - numeric) < tol, (
            f"input {i}: analytic={analytic:.6f}, numeric={numeric:.6f}"
        )

def _name(exc):
    #Helper to print exception names nicely, including tuples of types.def
    if isinstance(exc, tuple):
        return " | ".join(e.__name__ for e in exc)
    return exc.__name__

def assert_raises(exc, fn):
    #Minimal replacement for pytest.raise so this file runs without pytest.
    try:
        fn()
    except exc:
        return          #expected exception: OK
    except Exception as other:
        #A different exception was raised: not what we expected.
        raise AssertionError(
             f"expected {_name(exc)}, got {type(other).__name__}: {other}"
        )
    #If we get here, no exception was raise at all: fail.AssertionError
    raise AssertionError(f"expected {_name(exc)}, nothing was raised")


#Each test calls check() with a lambda that builds the expression and a set of input numbers.
# check() handles the gradient comparison.

def test_add():
    check(lambda a, b: a + b, 2.0, -3.0)
    
def test_mul():
    check(lambda a, b: a * b, 2.0, -3.0)

def test_sub():
    check(lambda a, b: a - b, 2.0, -3.0)

def test_div():
    check(lambda a, b: a / b, 2.0, -3.0)

def test_pow_integer():
    check(lambda a: a ** 3, 1.7)          # unary op: single input

def test_pow_fractional():
    check(lambda a: a ** 0.5, 4.0)        # base must be positive

def test_pow_negative_base_integer_exponent():
    check(lambda a: a ** 3, -2.0)

def test_pow_negative_base_even_exponent():
    check(lambda a: -a, 3.0)


#These test Python's __r*__ methods: when the left operand is an int and the right is a Value, Python
#falls back to Value.__r*__.

def test_add():
    a = Value(3.0)
    out = 2.0 + a           #calls a.__radd__(2.0)
    out.backward()
    assert a.grad == 1.0    #d(2+a)/da = 1

def test_rmul():
    a = Value(3.0)
    out = 2.0 * a           #calls a.__rmul__(2.0)
    out.backward()
    assert a.grad == 2.0        #d(2a)/da = 2

def test_rsub():
    a = Value(3.0) 
    out = 2.0 - a           #calls a.__rsub__(2.0)
    out.backward()
    assert a.grad == -1.0    #d(2-a)/da = -1

def test_rtruediv():
    a = Value(4.0)
    out = 2.0/a        #calls a.__rtruediv__(2.0)
    out.backward()
    #d(2/a) / da = -2/a^2 = -2/16 = -0.125
    assert abs(a.grad - (-0.125)) < 1e-6


#Now we do the same with non-linear functions

def test_tanh():
    check(lambda a: a.tanh(), 0.5)

def test_exp():
    check(lambda a: a.exp(), 0.5)

def test_log():
    check(lambda a: a.log(), 2.0)          # input must be > 0

def test_relu_positive():
    check(lambda a: a.relu(), 1.5)         # in the linear region

def test_relu_negative():
    check(lambda a: a.relu(), -1.5)        # derivative should be 0

def test_relu_at_zero():
    x = Value(0.0)
    y = x.relu()
    y.backward()
    assert y.data == 0.0                   # relu(0) = 0
    assert x.grad == 0.0                   # derivative at 0 is 0

#These check that += in _backward correctly accumulates gradients when the same node
# feeds multiple downstream paths.

def test_shared_node_a_times_a():
    a = Value(3.0)
    c = a * a                        # a appears twice as a parent
    c.backward()
    # d(a^2)/da = 2a = 6. If _backward used `=` instead of `+=`, this
    # would be 3 (only the second path) instead of 6.   
    assert abs(a.grad - 6.0) < 1e-6

def test_share_node_mixed():
    a = Value(3.0)
    c = a * a + a                   # a feeds two different ops
    c.backward()
    # d(a^2 + a)/ da = 2a + 1 = 7
    assert abs(a.grad - 7.0) < 1e-6


#Compositions

def test_composition():
    #A larger expression mixing arithmetic and nonlinear ops.test_composition
    #Finite-difference check against each of the two imputs.
    check(
        lambda a, b: (a * b + a.tanh()) / (b + 2.0) - a ** 2, 1.5 , 0.5,
    )

def test_deep_composition():
    #Nested ops: long(exp + tanh * square). Ensures the chain rule composoes correclty across several layers
    #of the graph.OverflowError
    check(lambda x: (x.exp() + x.tanh() * x ** 2).log(), 0.5,)

#Edge cases and error handling

def test_pow_negative_base_fractional_raises():
      # (-2)^0.5 would be a complex number in Python. __pow__ must refuse.
      assert_raises(ValueError, lambda: Value(-2.0) ** 0.5)

def test_log_nonpositive_raises():
    #log(0) and log(negative) are undefined. Both must raise. 
    assert_raises((AssertionError,ValueError), lambda: Value(0.0).log())
    assert_raises((AssertionError,ValueError), lambda: Value(-1.0).log())

def test_exp_overflow_raises():
    assert_raises(OverflowError, lambda: Value(800.0).exp())

#Autograd semantics

def test_backward_seeds_output():
    #backward() must set the output's own grad to 1.0 before traversing.
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert a.grad == 0.0          #untouched before backward
    c.backward()
    assert c.grad == 1.0          # seeded by backward


def test_grad_accumulates():
    #Gradients accumulate accros backward() calls ( PyTorch semantics). 
    a = Value(2.0)
    b = Value(3.0)
    c1 = a * b
    c1.backward()
    assert a.grad == 3.0
    c2 = a * b
    c2.backward()                   #no zero_grad in between
    assert a.grad == 6.0            # 3 + 3

def test_zero_grad():
    #zero_grad() must reset gradients on every node reachable from self.
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    assert a.grad != 0.0                   # something was accumulated
    c.zero_grad()
    assert a.grad == 0.0
    assert b.grad == 0.0
    assert c.grad == 0.0

#runner

if __name__ == "__main__":
    # Collect every function whose name starts with "test_".
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]

    # Run them one by one, printing PASS or FAIL for each.
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            failed += 1

    # Final summary line: "N passed, M failed".
    print(f"\n{passed} passed, {failed} failed")