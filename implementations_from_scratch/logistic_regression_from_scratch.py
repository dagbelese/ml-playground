# Import libraries
import numpy as np

class LogisticRegression():
    """
    Implementation of Logistic Regression model.
    """
    def __init__(self, learning_rate = 0.01, iterations = 1000):
        """
        Initialize Logistic Regression model.
        
        Args:
            learning_rate: Step size for gradient descent updates.
            iterations: Number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def fit(self, X, Y):
        """
        Train Logistic Regression model using gradient descent.

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
            self.update_weight()
            loss = self.compute_loss()
            self.losses.append(loss)

            # Convergence criterion
            if np.all(np.abs(self.W - prev_W) < 1e-06) and np.abs(self.b - prev_b) < 1e-6:
                print(f"Converged after {i+1} iterations.")
                break
    
        return self
    
    def update_weight(self):
        """
        Perform single gradient step to upgrade weights and bias
        """
        probabilities = self.predict_proba(self.X)
        error = probabilities - self.Y
        dW = self.X.T.dot(error) / self.num_samples
        db = np.sum(error) / self.num_samples
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Compute probabilities using trained Logistic Regression model.
        """
        if not hasattr(self, 'W') or not hasattr(self, 'b'):
            raise ValueError("Model has not been trained. Call 'fit' before 'predict_proba'.")
        probabilities = 1 / (1 + np.exp(- (X.dot( self.W ) + self.b)))
        return probabilities
    
    def predict(self, X):
        """
        Make predictions using trained Logsitic Regression model.
        """
        probabilities = self.predict_proba(X)
        predictions = np.where(probabilities > 0.5, 1, 0)
        return predictions
    
    def compute_loss(self):
        """
        Compute log loss.
        """
        probabilities = self.predict_proba(self.X)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15) # Avoid log(0) probabilities
        loss = -np.mean(self.Y * np.log(probabilities) + (1 - self.Y) * np.log(1 - probabilities))
        return loss
        