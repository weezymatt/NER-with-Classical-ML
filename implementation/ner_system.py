from abc import ABC, abstractmethod
import argparse
import pdb
import math
import collections
import numpy as np
import corpus
# from models import LogisticRegression as model # PENDING LOGISTIC REGRESSION from scratch 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

# TODO: LOGISTIC REGRESSION AND STUFF


class NERecognition(ABC):
	@abstractmethod
	def __init__(self, sentences):
		pass

	@abstractmethod
	def label(self, sentences):
		pass

class AlwaysNonEntity(NERecognition):
	""" 
	This a naive baseline that labels all token as non-entities. This metric demonstrates
	that token-level accuracy is not useful. The classic F-1 measure is also skewed by 
	non-entities. Instead exact-match (a spin on F-1) will be used which only considers
	named entities.

	The difficulty of the NER task can be hypothesized with the AlwaysNonEntity and
	SingleEntity baseline to determine how many entities are present.
	"""
	def __init__(self, sentences):
		self.default = 'O'

	def label(self, sentences):
		'''
		Function for labeling the token attributes with their respective ne.

		@params sentences: dataset

		@return: None
			* correctly tagged attributes of each token
		'''
		for sentence in sentences:
			string = sentence[1:-1]
			for token in string:
				token.ne = self.default

class SingleEntity(NERecognition):
	""" This is a less naive baseline system that labels identifies entities based on
		a lookup table. Only the beginning of entities are added to the table and
		the inside I-"<NE>" is to be ignored.
		
		(e.g., Jason [B-PER], Camacho [I-PER])

		which had a unique class in the training data. If a phrase
		contains more than one entity, the longest one is chosen. 
	"""
	def __init__(self, sentences):
		self.entities = collections.defaultdict(str)
		self.default = 'O'

		self._create_lookup(sentences)


	def _create_lookup(self, sentences):
		"""
		Helper method to create a dictionary of single named entities. The inside
		sequence is ignored.

		@params sentences: dataset

		@return: dictionary of ne
		"""
		for sentence in sentences:
			for token in sentence:
				if token.ne.startswith(('B-')):
					self.entities[token.word] = token.ne
		
	def label(self, sentences):
		for sentence in sentences:
			string = sentence[1:-1]
			for token in string:
				if self.entities.get(token.word, ''):
					token.ne = self.entities[token.word]
				else:
					token.ne = self.default

class FirstOrderHMM(NERecognition):
	""" This is a first order hidden Markov model that is designed to be language
		independent. That is, the machine learning model does not employ any
		language specific rules for handling unknown words.

		* Additive smoothing (alpha=100) performs decent.
		* Other smoothing techniques were not applied.
		* Unsure if a second order HMM would be better. More better would be crucial.
	"""
	def __init__(self, sentences):
		self.labels = collections.defaultdict(int)
		self.transition_counts = collections.defaultdict(lambda: collections.defaultdict(int))
		self.emission_counts = collections.defaultdict(lambda: collections.defaultdict(int))
		self.transitions = collections.defaultdict(lambda: collections.defaultdict(float))
		self.emissions = collections.defaultdict(lambda: collections.defaultdict(float))
		self.vocabulary = set()
		self._compute_probabilities(sentences)

	def _compute_probabilities(self, sentences):
		'''
		Helper method for constructing the probability lookup tables for the transitions and 
		emissions. For each transition and emission dictionary, it is based on the previous 
		unigrams (counts of single tokens). The vocabulary is also constructed.

		@params sentences: dataset

		@return: None
			* self.transitions
			* self.emissions
			* self.vocabulary
		'''

		for sentence in sentences:
			for i in range(len(sentence)):
				if i < len(sentence):
					self.vocabulary.add(sentence[i].word)
					self.labels[sentence[i].ne] += 1
					self.emission_counts[sentence[i].ne][sentence[i].word] += 1
				if i < (len(sentence) - 1):
					self.transition_counts[sentence[i].ne][sentence[i+1].ne] += 1

		for prev_label in self.labels:
			for next_label in self.labels:
				self.transitions[prev_label][next_label] = math.log((self.transition_counts[prev_label][next_label]+100.1)/(self.labels[prev_label]+ 100.1 * len(self.labels)),10)

		for label in self.labels:
			for word in self.vocabulary:
				if self.emission_counts[label] and word in self.emission_counts[label]:
					self.emissions[label][word] = math.log((self.emission_counts[label][word]/self.labels[label]), 10)

		self.start = self.transitions['<s>']
		del self.labels['<s>'] #delete </s>? does it matter? No. Ultimately we don't bother with this token to predict it!

	def _viterbi(self, s):
		'''
		Algorithm for decoding the most likely tag sequence. The process solves the NER task
		by using dynamic programming, that is, it breaks down the task into subtasks by means
		of a matrix.

		@params s: sentence

		@returns tag sequence: sequence of decoded tags
		'''
		grid = np.zeros([len(self.labels), len(s)])
		best = np.zeros([len(self.labels), len(s)], dtype=int)

		for wpos in range(len(s)):
			if wpos == 0:
				prev = self.start
			else:
				prev = grid[:, wpos-1]

			for i, label in enumerate(self.labels.keys()):
				if wpos == 0:
					if label in prev and s[wpos].word in self.emissions[label]:
						grid[i, wpos] = prev[label] + self.emissions[label][s[wpos].word]
					else:
						grid[i, wpos] = -1e6
				else:
					val = []
					for j, curr_label in enumerate(self.labels.keys()):
						curval = prev[j]
						curval += self.transitions[curr_label][label]
						if s[wpos].word in self.emissions[label]:
							curval += self.emissions[label][s[wpos].word]
						else:
							curval += -1e6
						val.append(curval)
					grid[i][wpos] = (max(val))
					best[i][wpos] = val.index(max(val))
					# pdb.set_trace()
		last_col = grid.argmax(axis=0)[-1]
		res = [last_col]
		i = len(s)-1
		while i >= 1:
			res.append(best[res[-1], i])
			i -= 1
		res.reverse()
		return res 

	def label(self, sentences):
		for sentence in sentences:
			string = sentence[1:-1]

			best_path = self._viterbi(string)

			for i, token in enumerate(string):
				for j, label in enumerate(self.labels):
					if j == best_path[i]:
							token.ne = label

class MEMM(NERecognition):
	'''
	A Maximum Entropy Markov Model (MEMM) a discriminative classifier that is powered 
	by Logistic Regression (LR). Naively, the model can be understood as applying LR
	to each token and using the current configuration as features.
	'''
	def __init__(self, sentences):
		self.label_encoder = LabelEncoder()
		self.feature_encoder = DictVectorizer()
		self.vocabulary = set()

		# self.model = model.LogisticRegression(solver="newton-raphson", max_iters=1000) # can write LR using schotchastic gradient descent
		self.model = LogisticRegression(penalty='l2', C=10., max_iter=500)
		self.train(sentences)

	def _extract_features(self, token, sentence, i, prev_feats):
		"""
		Method for extracting features for each token in the NER dataset. The matrix
		is implicitly constructed and can be understood as each row containing a token
		and a set of attributes that define its configuration.

		* X is a NxF matrix; N = total # of words, F = total # of features

		@param token: word
		@param sentence: sentence used for extra features
		@param index: index for accessing these extra features

		@return: dictionary of features
		"""
		word = token.word
		prev_label, prev_pos, prev_word = prev_feats

		features = {
				'len_sent': len(sentence),
				'isFirst': 1 if i == 0 else 0,
				'token': token.word, 
				'prev_label': prev_label,
				'curr_pos': token.pos,
				'prev_pos': prev_pos,
				'prev_word': prev_word,
				'isUpper': token.word[0].isupper(),
				'allUpper': token.word.isupper(), 
				'length': len(token.word),
				'numeric': token.word.isnumeric(),
				'feats': token.chunk,
				'index': i,
				'has_hyphen': '-' in word, 
				'alphanum': token.word.isalnum(),
				# '2_letters': 1 if len(word) == 2 else 0,
				'6_letters': 1 if len(word) > 6 else 0,
				'proper_noun': 1 if token.pos.startswith('NNP') else 0,
				'prev_The': 1 if prev_word.lower().startswith('the') else 0
		}
		return features

	def train(self, sentences):
		"""
		Overloaded method that both creates the matrix and trains the MEMM
		classifier on NER labeled sentences. The method also separates the 
		labels into a vector.

		@param sentences: dataset 

		@return: None
			* self.X
			* self.y
			* self.model.fit(dataset)
		"""
		matrix = []
		targets = []

		for sentence in sentences:
			for i, token in enumerate(sentence[1:-1]):
				if i == 0:
					prev_label = '<s>'
					prev_pos = '<s>'
					prev_word = '<s>'
				else:
					prev_label = sentence[i-1].ne
					prev_pos = sentence[i-1].pos
					prev_word = sentence[i-1].word

				prev_feats = tuple((prev_label, prev_pos, prev_word))

				features = self._extract_features(token, sentence, i, prev_feats)
				matrix.append(features)
				targets.append(token.ne)

		assert len(matrix) == len(targets)

		self.feature_encoder.fit(matrix)
		self.label_encoder.fit(targets)

		self.X = self.feature_encoder.transform(matrix)
		self.y = self.label_encoder.transform(targets)

		self.model.fit(self.X, self.y)

	def predict(self, sentence):
		'''
		Simple method to decode the predicted sequence given by the 
		algorithm. Greedy decoding is the default option since it is
		the quickest.

		@params sentence:  sentence

		@return ne_labels: tag sequence of ne labels
		'''
		# ne_labels = self.greedy_sequence_decoding(sentence)
		ne_labels = self.viterbi(sentence)
		return ne_labels

	def greedy_sequence_decoding(self, sentence):
		'''
		Greedy sequence decoding chooses the maximized label from left to right. This 
		type of decoding performs a hard decision and does not rely on the various
		configurations for each decision. 

		@params sentence: string of words

		@return tag sequence: sequence of ne labels
		'''
		tag_sequence = []

		for i, token in enumerate(sentence[1:-1]):
			if i == 0:
				prev_label = '<s>'
				prev_pos = '<s>' 
				prev_word = '<s>'
			else:
				prev_label = sentence[i-1].ne 
				prev_pos = sentence[i-1].pos
				prev_word = sentence[i-1].word

			prev_feats = tuple((prev_label, prev_pos, prev_word))

			feats = self._extract_features(token, sentence, i, prev_feats)

			encoded_feats = self.feature_encoder.transform(feats)

			encoded_prediction = self.model.predict(encoded_feats)

			decoded_prediction = self.label_encoder.inverse_transform(encoded_prediction)[0]

			tag_sequence.append(decoded_prediction)

		return tag_sequence

	def _get_probabilities(self, features):
		''' 
		Helper function for Viterbi. PENDING
		return label probabilities
		'''
		probs = self.model.predict_proba(features)
		# pdb.set_trace()
		log_probs = np.log10(probs.flatten())
		# pdb.set_trace()
		return log_probs

	def viterbi(self, s):
		# see textbook. Seems ok actually.
		# pending NOT SURE YET!
		# on average completes a matrix in 1 second 
		s = s[1:-1]
		grid = np.zeros([len(np.unique(self.y)), len(s)])
		best = np.zeros([len(np.unique(self.y)), len(s)], dtype=int)

		for wpos in range(len(s)):
			if wpos != 0:
				prev = grid[:, wpos-1]

			# for i, label in enumerate(np.unique(self.y)):
			for i, label in enumerate(self.label_encoder.classes_):
				# pdb.set_trace()
				if wpos == 0:
					for i, label in enumerate(np.unique(self.y)):
						prev_label = "<s>"
						prev_word = "<s>"
						prev_pos = "<s>"
						prev_feats = tuple((prev_label, prev_word, prev_pos))
						feats = self._extract_features(s[wpos], s, i, prev_feats)
						encoded_feats = self.feature_encoder.transform(feats) 
						probs = self._get_probabilities(encoded_feats)
						# pdb.set_trace()
						grid[i, wpos] = probs[label]
				else:
					val = []
					for j, curr_label in enumerate(np.unique(self.y)):
						prev_label = curr_label # none of these make sense yet! pending
						prev_word = s[wpos-1].word
						prev_pos = s[wpos-1].pos

						prev_feats = tuple((prev_label, prev_word, prev_pos))

						curval = prev[j]
						feats = self._extract_features(s[wpos], s, i, prev_feats) # verify this is the right previous label
						encoded_feats = self.feature_encoder.transform(feats)
						probs = self._get_probabilities(encoded_feats) # are these log probabilities?
						# pdb.set_trace()
						curval += probs[j]
						val.append(curval)
					grid[i][wpos] = (max(val))
					best[i][wpos] = val.index(max(val))
					# pdb.set_trace()
		last_col = grid.argmax(axis=0)[-1]
		res = [last_col]
		i = len(s)-1
		while i >= 1:
			res.append(best[res[-1], i])
			i -= 1
		res = self.label_encoder.inverse_transform(res)
		return res 

	def label(self, sentences):
		for k, sentence in enumerate(sentences):
			labels = self.predict(sentence)
			# labels = self.label_encoder.inverse_transform(labels)
			# pdb.set_trace()
			for i, token in enumerate(sentence[1:-1]):
				token.ne = labels[i]
			# pdb.set_trace()
			print(f"{k} sentence DONE!")
