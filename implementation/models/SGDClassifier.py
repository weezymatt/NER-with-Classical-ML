import numpy as np 
import pdb
import time
from scipy.sparse import csr_matrix, issparse

# testing purposes
from sklearn import datasets

class SGDClassifier:
	def __init__(self, max_iters=300, eta0=0.005, C=0.0001, random_state=42):
		self.max_iters = max_iters # epochs
		self.eta0 = eta0 # constant learning rate
		self.C = C # strength of regularization
		self.losses = []
		self.random_state = random_state

	def fit(self, X, y): 
		classes = np.unique(y)
		W = np.zeros([len(classes), X.shape[1]], dtype=np.float32) 
		b = np.ones([len(classes), 1], dtype=np.float32) 
		self.fit_data(X, y, W, b, classes)
		return self

	def fit_data(self, X, y, W, b, classes): 
		if self.random_state:
			np.random.seed(self.random_state)

		y_one_hot = np.eye(len(classes))[y]

		for epoch in range(self.max_iters):
			indices = np.random.permutation(X.shape[0])
			X = X[indices] #new
			y_one_hot = y_one_hot[indices] #new
			i = 0
			loss_epoch = []
			while i < X.shape[0]: 
				x = X[i] 

				logits = W @ x.T + b 

				y_pred = self.softmax(logits)

				y_true = y_one_hot[i]

				loss = self.cross_entropy(y_true, y_pred, W)

				loss_epoch.append(loss) 

				grad_W = self.gradient_W(x, y_pred, y_true, W)

				grad_b = self.gradient_b(y_pred, y_true)

				W -= self.eta0 * grad_W
				b -= self.eta0 * grad_b

				i += 1

			avg_loss = np.sum(loss_epoch) / X.shape[0]

			self.losses.append(avg_loss)
			print(f"Epoch: {epoch+1} completed! Loss: {self.losses[-1]:.2f}") 
			# Regularization results in higher loss. Loss works but see more later.
			# Fixed. Just multiply alpha or C to the regularization term
		self.W = W 
		self.b = b

	def predict(self, X): 
		logits_vec = self.W @ X.T + self.b
		y_pred = self.softmax(logits_vec, train=False)
		y = np.argmax(y_pred, axis=0) 
		return y.tolist()

	def gradient_W(self, x, y_pred, y_true, W):
		Err = y_pred - y_true
		Err = Err.reshape(-1, 1)
		grad = Err @ x + self.C * 2 * W
		return grad

	def gradient_b(self, y_pred, y_true):
		Err = y_pred - y_true
		return Err.reshape(-1,1)

	def cross_entropy(self, y_true, y_pred, W):
		idx = np.argmax(y_true) 
		epsilon = 1e-8
		L2_term = self.C * np.sum(np.power(W, 2)) #rewrite for 1/2
		CE_term = -(np.log(y_pred[idx] + epsilon))
		return CE_term + L2_term

	def softmax(self, z, train=True):
		z = z - np.max(z)
		exp_z = np.exp(z)  
		partition = np.sum(exp_z)
		if train:
			return (exp_z / partition).flatten()
		else: 
			return (exp_z / partition)

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
