import math
from engine import Value

def numeric_grad(f, values, idx, eps=1e-6):
    """
    Compute the partial derivate of f w.r.t values[idx] via central differences.
    `f` takes a list of Values and returns a Value.
    We rebuild the graph twice(with +eps and -eps) so we don't pollute the original
    """
    data_orig = [v.data for v in values]

    #-eps
    plus = [Value(d) for d in data_orig]
    plus[idx].data += eps
    f_plus = f(plus).data

    #-eps