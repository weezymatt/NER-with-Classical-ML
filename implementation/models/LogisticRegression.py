import numpy as np 
import pdb
import time
from scipy.sparse import csr_matrix, issparse

# testing purposes
from sklearn import datasets

class SGDClassifier:
	def __init__(self, max_iters=200, eta0=0.05, C=0.00001):
		self.max_iters = max_iters # epochs
		self.eta0 = eta0 # constant learning rate
		self.C = C # strength of regularization

	def fit(self, X, y): # ensure all dimensions are correct
		self.classes = np.unique(y)
		self.W = np.zeros([len(self.classes), X.shape[1]]) # (num_classes, num_features)
		self.b = np.ones([len(self.classes), 1]) # (num_classes, 1)
		self.losses = []
		self.fit_data(X, y)
		return self

	def fit_data(self, X, y): 
		y_one_hot = np.eye(len(self.classes))[y]

		for epoch in range(self.max_iters):
			i = 0
			loss_epoch = []
			while i < X.shape[0]: 
				if issparse(X):
					X = X.todense()
				x = X[i]
				logits = self.W @ x.reshape(-1 ,1) + self.b 
				y_pred = self.softmax(logits)

				y_true = y_one_hot[i]

				loss = self.cross_entropy(y_true, y_pred)

				loss_epoch.append(loss) 

				ERR = -(y_true - y_pred.flatten())

				grad = ERR.reshape(-1, 1) @ x.reshape(-1, 1).T + (self.C * 2 * self.W)

				self.W = self.W - self.eta0 * grad 
				self.b = self.b - self.eta0 * ERR.reshape(-1, 1)
				i += 1

			avg_loss = np.sum(loss_epoch) / X.shape[0]

			self.losses.append(avg_loss)
			print(f"Epoch: {epoch+1} completed! Loss: {self.losses[-1]:.2f}") 
			# Regularization results in higher loss. Loss works but see more later.

	def predict(self, X): #rewrite here
		logits_vec = self.W @ X.T + self.b
		y_pred = self.softmax(logits_vec)
		y = np.argmax(y_pred, axis=0) 
		return y.tolist()[0]

	def cross_entropy(self, y_true, y_pred):
		idx = np.argmax(y_true) # find the correct class
		epsilon = 1e-8
		L2_term = np.sum(np.power(self.W, 2))
		y_pred = np.array(y_pred).flatten() # simplified cross entropy fn
		CE_term = -(np.log(y_pred[idx] + epsilon))
		return CE_term + L2_term

	def softmax(self, z):
		z = z - np.max(z, axis=0) # numerically stable
		exp_z = np.exp(z) 
		partition = np.sum(exp_z)
		return exp_z / partition


# python3 LogisticRegression.py
if __name__ == '__main__':
	data = datasets.load_iris()
	X = csr_matrix(data['data'])
	y = data['target']
	# a = time.time()
	model = SGDClassifier()
	model.fit(X,y)
	# b = time.time()
	predictions = model.predict(X) # just a quick test
	accuracy = np.mean(predictions == y)
	print(f"Accuracy: {accuracy * 100:.2f}%")
	# print(b-a)
