class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def predict(self, X):
        """
        predict the outputs for n number of inputs 
        
        :param X: all the test inputs (vector)
        """
        results = []

        for x in X:
            output = x
            for layer in self.layers:
                output = layer.forward(output)
            
            results.append(output)

        return results
    
    def train(self, loss_func, loss_prime, X_train, y_train, epochs=100, lr=0.001):
        for epoch in range(epochs):
            error = 0

            for x, y in zip(X_train, y_train):

                # perform the forward pass
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                
                error += loss_func(y, output)

                # loss will the error say E
                # dE / d_activations = loss prime 
                grad = loss_prime(y, output)

                for layer in reversed(self.layers):
                    # take loss form each layer and pass it to the previous one 
                    grad = layer.backward(grad, lr)
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}  error={error/len(X_train)}')
