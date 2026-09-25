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