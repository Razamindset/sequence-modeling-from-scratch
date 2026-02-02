class Optimizer:
    def __init__(self, params, grads):
        raise NotImplementedError
    

# Update for each parameter seprately 
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, gradients):
        for key in params:
            params[key] -= self.lr * gradients[key]

        return params
