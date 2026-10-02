class SGD:
    """
    Stochastic Gradient Descent.
    params: list of Value objects to update
    lr: learning rate
    """
def __init__(self, params, lr = 0.01):
    self.params = list(params)
    self.lr = lr
