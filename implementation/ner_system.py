from abc import ABC, abstractmethod
import argparse
import pdb
import math
import collections
import numpy as np

import corpus
# from models import LogisticRegression as model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
# https://www.clips.uantwerpen.be/conll2002/ner/

class NERecognition(ABC):
	@abstractmethod
	def __init__(self, sentences):
		pass

	@abstractmethod
	def label(self, sentences):
		pass

class AlwaysNonEntity(NERecognition):
	""" This a weak baseline that labels all token as non-entities. The metric demonstrates
		that token-level accuracy nor F1-score is useful for the NER task.
	
	English:
		- Validation set
			accuracy: 0.8325026284023208
			exact-match: 0.0
		- Test set
			accuracy: 82.53041886508022
			exact-match: 0.0

	German:
		- Validation set
			accuracy:
			exact-match:
		- Test set
			accuracy:
			exact-match

		German accuracy: 90.04485686232985 PENDING! Numbers are off.
	"""
	def __init__(self, sentences):
		self.default = 'O'

	def label(self, sentences):
		for sentence in sentences:
			string = sentence[1:-1]
			for token in string:
				token.ne = self.default

class SingleEntity(NERecognition):
	""" This is a better baseline system that only identifies entities
		which had a unique class in the training data. If a phrase
		contains more than one entity, the longest one is chosen. 

		English:
			- Validation set
				accuracy:  0.8653284529418637
				exact-match: 0.4009203737275136

			- Test set
				accuracy:  0.8488640034456768
				exact-match: 0.3539166140240051

		German:
			- Validation set
				accuracy:
				exact-match

			- Test set
				accuracy:
				exact-match:

		German:
			accuracy: 0.9004485686232986 possible problem and not getting also entities!!!
			exact-match: 0.014496644295302013

	"""
	def __init__(self, sentences):
		self.entities = collections.defaultdict(str)
		self.default = 'O'

		self._create_lookup(sentences)


	def _create_lookup(self, sentences):
		for sentence in sentences:
			for token in sentence:
				if token.ne.startswith(('B-')):
					self.entities[token.word] = token.ne
		
	def label(self, sentences):
		# pdb.set_trace()
		for sentence in sentences:
			string = sentence[1:-1]
			for token in string:
				if self.entities.get(token.word, ''):
					token.ne = self.entities[token.word]
				else:
					token.ne = self.default

class FirstOrderHMM(NERecognition):
	""" This is a first-order hidden Markov model that is designed to be language
		independent (i.e., it does not employ language-specific rules for handling
		unknown words). 

		SIDE NOTE: why smooth the transitions and not emssions? tag -> tag is important! but emissions not really because every word has a label? doesn't need smoothing.

		English:
			- Validation set
				accuracy: 0.9512480043612009
				exact-match: 0.7901645082254113

			- Test set
				accuracy: 0.9253580273500592
				test exact-match: 0.6762441314553991

		German:
			- Validation set
				accuracy:
				exact-match

			- Test set	
				accuracy: pending possible problem
				exact-match: pending
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
				self.transitions[prev_label][next_label] = math.log((self.transition_counts[prev_label][next_label]+.1)/(self.labels[prev_label]+ .1 * len(self.labels)),10)

		for label in self.labels:
			for word in self.vocabulary:
				if self.emission_counts[label] and word in self.emission_counts[label]:
					self.emissions[label][word] = math.log((self.emission_counts[label][word]/self.labels[label]), 10)

		self.start = self.transitions['<s>']
		del self.labels['<s>'] #delete </s>? does it matter? No. Ultimately we don't bother with this token to predict it!

	def _viterbi(self, s):
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

class SecondOrderHMM(NERecognition): # maybe not... 
	def __init__(self, sentences):
		self.labels = collections.defaultdict(int) # C(t_i-1)
		self.transition_counts = collections.defaultdict(lambda: collections.defaultdict(int))
		self.transitions = collections.defaultdict(lambda: collections.defaultdict(float))
		self.bigram_counts = collections.defaultdict(lambda: collections.defaultdict(int))
		self.emission_counts = collections.defaultdict(lambda: collections.defaultdict(int)) 
		self.emissions = collections.defaultdict(lambda: collections.defaultdict(float))
		self.vocabulary = set()

		self._pad(sentences)
		self._compute_probabilities(sentences)

	def _compute_probabilities(self, sentences):
		for sentence in sentences:
			for i in range(len(sentence)):

				word = sentence[i].word
				prev_2 = sentence[i].ne

				if i < len(sentence):
					self.vocabulary.add(word)
					self.labels[prev_2] += 1
					self.emission_counts[prev_2][word] += 1
				if i < len(sentence) - 2:
					prev_1 = sentence[i+1].ne
					self.bigram_counts[prev_2][prev_1] += 1
				if i < (len(sentence) - 2):
					curr = sentence[i+2].ne

					self.transition_counts[tuple((prev_2, prev_1))][curr] += 1

		# del self.labels['</s>']

		for prev_2 in self.labels:
			for prev_1 in self.labels:
				history = tuple((prev_2, prev_1))
				for curr in self.labels:
					self.transitions[history][curr] = math.log10((self.transition_counts[history][curr] + 1.0)/(self.bigram_counts[prev_2][prev_1] + len(self.labels) * 1.0))

		for label in self.labels:
			for word in self.vocabulary:
				if self.emission_counts[label] and word in self.emission_counts[label]:
					self.emissions[label][word] = math.log10(self.emission_counts[label][word]/self.labels[label])

		self.start = self.transitions[tuple(('<s>', '<s>'))]
		del self.start['<s>']
		del self.start['</s>']
		del self.transitions[('<s>','<s>')]
		del self.transitions[('<s>', '</s>')]

		self.transitions = {k: v for k, v in self.transitions.items() if k[1] != '<s>'}
		self.transitions = {k: v for k, v in self.transitions.items() if k[0] != '</s>'}

	def _pad(self, sentences):
		for sentence in sentences:
			sentence.insert(0, corpus.Token('<s>', '<s>', '<s>', '<s>'))
			sentence.append(corpus.Token('</s>', '</s>', '</s>', '</s>'))

	def _viterbi(self, s):
		grid = np.zeros([len(self.transitions), len(s)])
		best = np.zeros([len(self.transitions), len(s)], dtype=int)

		for wpos in range(len(s)):
			if wpos == 0:
				prev = self.start
			else:
				prev = grid[:, wpos-1]

			for i , historyi in enumerate(self.transitions.keys()): #?
				state_2, state_1 = historyi

				if wpos == 0:
					for j, historyj in enumerate(self.transitions.keys()):
						State_1, state_i = historyj
						if state_1 == State_1:
							if state_i in prev and s[wpos].word in s[wpos].word in self.emissions[state_i]:
								grid[i, wpos] = prev[state_i] + self.emissions[state_i][s[wpos].word]
						else:
							grid[i, wpos] = -1e6 #-1e6
				else:
					val = []
					for k, historyk in enumerate(self.transitions.keys()): 
						State_1, state_i = historyk

						if state_1 == State_1:
							curval = prev[k] 
							curval += self.transitions[historyi][state_i]
							if s[wpos].word in self.emissions[state_i]:
								# pdb.set_trace()
								curval += self.emissions[state_i][s[wpos].word]
							else:
								curval += -1e6
							val.append(curval)
						else:
							curval = -1e12
							val.append(curval)
					grid[i][wpos] = (max(val))
					best[i][wpos] = val.index(max(val))

		last_col = grid.argmax(axis=0)[-1]
		res = [last_col]
		i = len(s)-1
		while i >= 1:
			res.append(best[res[-1], i])
			i -= 1
		res.reverse()
		# pdb.set_trace()
		return res 

	def label(self, sentences):
		self._pad(sentences)

		for k, sentence in enumerate(sentences):
			string = sentence
			# pdb.set_trace()

			best_path = self._viterbi(string)

			for i, token in enumerate(string):
				for j, label in enumerate(self.transitions.keys()):
					# pdb.set_trace()
					if j == best_path[i]:
							token.ne = label[1]
			# pdb.set_trace()
			print(f"{k} sentence DONE!")
			# pdb.set_trace()

class MEMM(NERecognition):
	# TODO: write logistic regression first, then build the MEMM 
	def __init__(self, sentences):
		self.label_encoder = LabelEncoder()
		self.feature_encoder = DictVectorizer()

		# self.model = model.LogisticRegression(solver="newton-raphson", max_iters=1000) # can write LR using schotchastic gradient descent
		self.model = LogisticRegression(max_iter=500)
		self.train(sentences)

	def _extract_features(self, token, sentence, i):
		"""
		Method for extracting features for each token (row) in the NER dataset.

		@param token: word 
		@param sentence: sentence used for extra features
		@param index: index for accessing these extra features
		"""
		features = {
				'isFirst': 1 if i == 0 else 0,
				# 'isLast'
				'token': token.word,
				'token-1': sentence[i-1].word if i != 0 else 'None',
				'pos-1': sentence[i-1].pos if i != 0 else '<s>',
				'pos+1': sentence[i+1].pos if i < len(sentence) - 1 else '<s>',
				'pos': token.pos,
				'isUpper': token.word[0].isupper(),
				'allUpper': token.word.isupper(), #istitle
				'length': len(token.word),
				'numeric': token.word.isnumeric(),
				'feats': token.chunk,
				'has_hyphen': '-' in token.word, 
				'alphanum': token.word.isalnum(),
				'2_letters': 1 if len(token.word) == 2 else 0
		}
		return features

	def train(self, sentences):
		"""
		Trains the classifier on NER labeled sentences,
		and returns the feature matrix and label vector
		"""
		matrix = []
		targets = []

		for sentence in sentences:
			for i, token in enumerate(sentence[1:-1]):
				# if i != 0:
				features = self._extract_features(token, sentence, i)
				matrix.append(features)
				targets.append(token.ne)

		assert len(matrix) == len(targets)

		self.feature_encoder.fit(matrix)
		self.label_encoder.fit(targets)

		self.X = self.feature_encoder.transform(matrix)
		self.y = self.label_encoder.transform(targets)

		self.model.fit(self.X, self.y)

	def predict(self, sentence):
		ne_labels = self.greedy_decoding(sentence)
		return ne_labels

	def greedy_decoding(self, sentence):
		targets_new = []

		for i, token in enumerate(sentence[1:-1]):
			# if i != 0:
			feats = self._extract_features(token, sentence, i)

			encoded_feats = self.feature_encoder.transform(feats)

			encoded_prediction = self.model.predict(encoded_feats)

			decoded_prediction = self.label_encoder.inverse_transform(encoded_prediction)[0]

			targets_new.append(decoded_prediction)

		return targets_new

	def viterbi(self, sentence):
		# see textbook. Seems ok actually.
		pass

	def label(self, sentences):
		for sentence in sentences:
			labels = self.predict(sentence)
			for i, token in enumerate(sentence[1:-1]):
				token.ne = labels[i]
















