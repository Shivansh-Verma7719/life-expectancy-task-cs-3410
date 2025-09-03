import numpy as np

class LinearRegression:
    # Linear Regression class
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Here we impelement Gradient Descent
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            # dw = 1/n * (y_pred - y) * x
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))

            # db = 1/n * sum(y_pred - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            # w = w - lr * dw
            self.weights -= self.learning_rate * dw

            # b = b - lr * db
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # y = wx + b
        # w and x are vectors
        return np.dot(X, self.weights) + self.bias