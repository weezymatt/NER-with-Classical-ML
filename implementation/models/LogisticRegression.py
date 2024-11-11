# logistic regression (for MEMM)
from sklearn.datasets import load_iris
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pdb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression as LR2


# class LogisticRegression():
# 	def __init__(self, solver='newton-raphson', max_iters=100, tol=1e-6):
# 		self.max_iters = max_iters
# 		self.tol = tol

# 	def _softmax(self, Z):
# 		exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
# 		return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# 	def _log_likelihood(self, P, y):
# 		# y = y.reshape(-1, 1)
# 		return np.sum(y * np.log(P))

# 	def fit(self, X, y):
# 		N, D = X.shape
# 		ones_vec = csr_matrix(np.ones((N, 1)))
# 		X = hstack((ones_vec, X))
# 		classes = np.unique(y)
# 		W = np.random.randn(len(classes), D+1)
# 		y = y.reshape(-1, 1)


# 		for epoch in range(self.max_iters):
# 			Z = X @ W.T 
# 			P = self._softmax(Z)

# 			ll = self._log_likelihood(P, y) # works but why does that likelihood work like that?
# 			# pdb.set_trace()

# 			grad = self.gradient(X, P, y)
# 			hess = self.hessian(X, P)

# 			W_new = W - np.linalg.inv(H) @ gradient.T

# 			print(epoch)

# 			if np.linalg.norm(W_new - W) < self.tol:
# 				break

# 	def gradient(self, X, P, y):
# 		return X.T @ (P - y)

# 	# def hessian(self, X, P):
# 	# 	N, C = P.shape
# 	# 	H = np.zeros((X.shape[1], X.shape[1]))  # Hessian size (D+1) x (D+1)
# 	# 	for i in range(N):
# 	# 		p = P[i, :].reshape(-1, 1)  # (C, 1)
# 	# 		outer = p @ p.T  # (C, C)
# 	# 		pdb.set_trace()
# 	# 		H += X[i, :].T @ outer @ X[i, :]
# 	# 	return H

# 	# def hessian(self, X, P):
# 	# 	N, D = X.shape
# 	# 	C = P.shape[1]

# 	# 	# Compute the diagonal matrix for the Hessian calculation
# 	# 	Lambda = np.zeros((N, C))

# 	# 	for i in range(N):
# 	# 		for c in range(C):
# 	# 			Lambda[i, c] = P[i, c] * (1 - P[i, c])

# 	# 	# Hessian: X^T Lambda X
# 	# 	H = X.T @ (Lambda * X)
# 	# 	return H

# 	# def hessian(self, X, P):
# 	#     """
# 	#     Compute the Hessian matrix of the negative log-likelihood.
# 	#     X: (N, D) - Design matrix
# 	#     P: (N, C) - Predicted probabilities from softmax
# 	#     """
# 	#     N, D = X.shape
# 	#     C = P.shape[1]

# 	#     # Initialize the Hessian matrix as a sparse matrix
# 	#     H = np.zeros((D, D))

# 	#     # Compute the diagonal elements of Lambda matrix for each sample
# 	#     # Lambda is a diagonal matrix, so each sample i has a diagonal element for each class c
# 	#     for i in range(N):
# 	#         # Get the probability vector for sample i
# 	#         pi = P[i]

# 	#         # Create the diagonal elements for this sample's Lambda matrix
# 	#         # Lambda_ii = pi * (1 - pi) for each class c
# 	#         Lambda_i = np.diag(pi * (1 - pi))

# 	#         # Update Hessian: H = X^T * Lambda * X (sum over all samples)
# 	#         X_i = X[i, :].reshape(-1, 1)  # Column vector for sample i
# 	#         H += X_i @ Lambda_i @ X_i.T

# 	#     return H

# 	def hessian(self, X, P):
# 	    N, D = X.shape  # N: number of samples, D: number of features
# 	    C = P.shape[1]  # C: number of classes

# 	    H = np.zeros((D, D))  # Initialize Hessian matrix

# 	    # Loop over all samples
# 	    for i in range(N):
# 	        # Predicted probabilities for the i-th sample
# 	        pi = P[i]  # shape (C,)

# 	        # Create the diagonal matrix Lambda_i of size (C, C)
# 	        Lambda_i = np.diag(pi * (1 - pi))  # Diagonal elements: P_{ic} * (1 - P_{ic})

# 	        # Get the feature vector X_i for sample i (this is a row vector of shape (D,))
# 	        X_i = X[i, :].reshape(-1, 1)  # Ensure X_i is a column vector of shape (D, 1)
# 	        pdb.set_trace()
# 	        # Compute X_i^T * Lambda_i * X_i (this is a scalar)
# 	        hess_i = X_i.T @ Lambda_i @ X_i  # shape (1, 1) scalar value

# 	        # Add the scalar to the Hessian matrix (it adds up to a (D, D) matrix)
# 	        H += hess_i

# 	    return H


#     # def predict(self, X):
# 	# 	N = X.shape[0]
# 	# 	ones_vec = np.ones((N, 1))  # Bias column of ones
# 	# 	X = hstack((ones_vec, X))  # Add bias to design matrix

# 	# 	# Compute logits and softmax probabilities
# 	# 	Z = X @ self.W.T
# 	# 	P = self._softmax(Z)

# 	# 	# Return the predicted class (argmax of probabilities)
# 	# 	return np.argmax(P, axis=1)

# import numpy as np

# class LogisticRegression:
#     def __init__(self, learning_rate=0.0001, max_iters=10000):
#         self.learning_rate = learning_rate
#         self.max_iters = max_iters

#     def _softmax(self, Z):
#         exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
#         return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

#     def fit(self, X, y):
#         N, D = X.shape  # N: number of samples, D: number of features
#         C = len(np.unique(y))  # C: number of classes

#         # Initialize weights (D + 1 because we have a bias term)
#         self.W = np.random.randn(D + 1, C) * 0.01  # Random initialization

#         # Add bias term (X0 = 1)
#         X = np.hstack([np.ones((N, 1)), X])

#         y = y.reshape(-1, 1)

#         # Gradient Descent
#         for i in range(self.max_iters):
#             # Compute predictions using softmax
#             Z = X @ self.W
#             P = self._softmax(Z)

#             # Compute the gradient of the negative log-likelihood
#             gradient = X.T @ (P - y) / N  # Average over all samples

#             # Update weights using the gradient
#             self.W -= self.learning_rate * gradient

#             # Optionally print the loss for monitoring
#             if i % 100 == 0:
#                 loss = -np.mean(np.sum(y * np.log(P), axis=1))
#                 print(f"Iteration {i}: Loss = {loss}")

#     def predict(self, X):
#         N = X.shape[0]
#         X = np.hstack([np.ones((N, 1)), X])  # Add bias term
#         Z = X @ self.W
#         P = self._softmax(Z)
#         return np.argmax(P, axis=1)  # Return the class with highest probability

	class LogisticRegression:
	    def __init__(self, learning_rate=0.01, max_iters=1000):
	        self.learning_rate = learning_rate
	        self.max_iters = max_iters

	    def _softmax(self, Z):
	        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # For numerical stability
	        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

	    def fit(self, X, y):
	        N, D = X.shape  # N: number of samples, D: number of features
	        C = len(np.unique(y))  # C: number of classes

	        # One-hot encoding of the labels y
	        Y = np.zeros((N, C))
	        Y[np.arange(N), y] = 1

	        # Initialize weights (D + 1 for bias term, C classes)
	        self.W = np.random.randn(D + 1, C) * 0.01  # Xavier/He initialization

	        # Add bias term (X0 = 1)
	        X = np.hstack([np.ones((N, 1)), X])

	        # Gradient Descent
	        for i in range(self.max_iters):
	            # Compute predictions using softmax
	            Z = X @ self.W
	            P = self._softmax(Z)

	            # Compute the gradient of the negative log-likelihood
	            gradient = X.T @ (P - Y) / N  # Average over all samples

	            # Update weights using the gradient
	            self.W -= self.learning_rate * gradient

	    def predict(self, X):
	        N = X.shape[0]
	        X = np.hstack([np.ones((N, 1)), X])  # Add bias term
	        Z = X @ self.W
	        P = self._softmax(Z)
	        return np.argmax(P, axis=1)  # Return the class with the highest probability

if __name__ == '__main__':
	data = load_iris()
	X = data['data']
	y = data['target']
	le = LabelEncoder()
	y = le.fit_transform(y)
	model = LogisticRegression()
	model2 = LR2()
	model.fit(X, y)
	model2.fit(X,y)

	predictions = model.predict(X)

	# Compute accuracy: the percentage of correct predictions
	accuracy = np.mean(predictions == y)
	print(f"Accuracy: {accuracy * 100:.2f}%")
	pdb.set_trace()