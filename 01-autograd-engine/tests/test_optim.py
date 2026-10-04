import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Value
from optim import SGD

# Basic update


def test_sgd_step_subtracts_lr_times_grad():
    a = Value(3.0)
    a.grad = 1.0

    opt = SGD([a], lr=0.1)
    opt.step()

    # 3.0 - 0.1 * 1.0 = 2.9
    assert abs(a.data - 2.9) < 1e-9


def test_sgd_step_multiple_params():
    a = Value(1.0)
    b = Value(2.0)
    a.grad = 1.0
    b.grad = 2.0

    opt = SGD([a, b], lr=0.5)
    opt.step()

    assert abs(a.data - 0.5) < 1e-9   
    assert abs(b.data - 1.0) < 1e-9   


def test_sgd_zero_lr_does_nothing():
    a = Value(3.0)
    a.grad = 5.0

    opt = SGD([a], lr=0.0)
    opt.step()

    assert a.data == 3.0


def test_sgd_lr_is_public_and_mutable():
    a = Value(3.0)
    a.grad = 1.0

    opt = SGD([a], lr=0.1)
    opt.lr = 0.5
    opt.step()

    # 3.0 - 0.5 * 1.0 = 2.5
    assert abs(a.data - 2.5) < 1e-9


def test_sgd_zero_grad_resets():
    a = Value(1.0)
    b = Value(2.0)
    a.grad = 1.0
    b.grad = 2.0

    opt = SGD([a, b], lr=0.1)
    opt.zero_grad()

    assert a.grad == 0.0
    assert b.grad == 0.0


def test_sgd_step_does_not_reset_grad():
    # step() not touch .grad. This is the accumulation contract
    a = Value(1.0)
    a.grad = 1.0

    opt = SGD([a], lr=0.1)
    opt.step()

    assert a.grad == 1.0



def test_sgd_accepts_generator():
    # A generator is consumed on the first iteration if stored as-is
    # We convert to list in the constructor to avoid this.
    a = Value(1.0)
    b = Value(2.0)
    a.grad = 1.0
    b.grad = 1.0

    opt = SGD((v for v in [a, b]), lr=0.5)
    opt.step()   # first iteration would consume the generator

    assert abs(a.data - 0.5) < 1e-9
    assert abs(b.data - 1.5) < 1e-9

    opt.step()   # second iteration would see nothing if not stored as list

    assert abs(a.data - 0.0) < 1e-9
    assert abs(b.data - 1.0) < 1e-9



# Ordering contract: zero_grad -> backward -> step

def test_full_training_step_ordering():
    # Simulate the pattern used in train_moons.py.
    a = Value(2.0)
    b = Value(3.0)

    opt = SGD([a, b], lr=0.1)

    # 1. zero_grad
    opt.zero_grad()
    assert a.grad == 0.0
    assert b.grad == 0.0

    # 2. forward + backward
    loss = (a * b).tanh()   # some differentiable expression
    loss.backward()

    # 3. step
    a_before = a.data
    b_before = b.data
    opt.step()

    assert a.data != a_before or a.grad == 0.0
    assert b.data != b_before or b.grad == 0.0

if __name__ == "__main__":
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")