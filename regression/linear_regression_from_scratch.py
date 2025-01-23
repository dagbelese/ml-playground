# Import libraries
import numpy as np

class LinearRegression():
    """
    Implementation of Linear Regression using only numpy.
    """
    def __init__(self, learning_rate = 0.01, iterations = 1000):
        """
        Initialize Linear Regression model.
        
        Args:
            learning_rate: Step size for gradient descent updates.
            iterations: Number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
 
    def fit(self, X, Y):       
        """
        Train Linear Regression model using gradient descent.

        Args:
            X: Training data, NumPy array of shape (num_samples, num_features).
            Y: Target values, NumPy array of shape (num_samples,).

        Returns:
            self: Trained model
        """
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("X and Y must be NumPy arrays.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Number of samples in X and Y must match.")

        self.num_samples, self.num_features = X.shape
        self.W = np.zeros(self.num_features)
        self.b = 0
        self.X = X
        self.Y = Y
        self.losses = []

        for i in range(self.iterations):
            prev_W = self.W.copy()
            prev_b = self.b
            self.update_weights()
            loss = self.compute_loss()
            self.losses.append(loss)

            # Convergence criterion
            if np.all(np.abs(self.W - prev_W) < 1e-06) and np.abs(self.b - prev_b) < 1e-6:
                print(f"Converged after {i+1} iterations.")
                break

        return self
    
    def update_weights(self):
        """
        Perform single gradient step to upgrade weights and bias.
        """
        Y_pred = self.predict(self.X)
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.num_samples
        db = - (2 * np.sum(self.Y - Y_pred)) / self.num_samples
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
            
    def predict(self, X):
        """
        Make predictions using trained Linear Regression model.
        """
        if not hasattr(self, 'W') or not hasattr(self, 'b'):
            raise ValueError("Model has not been trained. Call 'fit' before 'predict'.")
        return X.dot(self.W) + self.b
    
    def compute_loss(self):
        """
        Compute Mean Squared Error (MSE) loss.
        """
        Y_pred = self.predict(self.X)
        return np.mean((self.Y - Y_pred) ** 2) / self.num_samples

    

    
