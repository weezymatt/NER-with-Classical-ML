from abc import ABC, abstractmethod
import argparse
import pdb
import math
import collections
import numpy as np
import corpus
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import gensim.downloader as api
import spacy
import LogisticRegression as lr 

class NERecognition(ABC):
	@abstractmethod
	def __init__(self, sentences):
		pass

	@abstractmethod
	def label(self, sentences):
		'''
		Function for labeling the token attributes with their respective named entity.

		@params sentences: dataset
		@return: None
		'''
		pass

class AlwaysNonEntity(NERecognition):
	""" 
	The AlwaysNonEntity is a naive baseline that labels all token as non-entities. This
	demonstrates that NER is largely an unbalanced task that where token-level accuracy
	is not useful.
	"""
	def __init__(self, sentences):
		self.default = 'O'

	def label(self, sentences):
		for sentence in sentences:
			string = sentence[1:-1]
			for token in string:
				token.ne = self.default

class SingleEntity(NERecognition):
	""" This is a less naive baseline system that labels identifies entities based on
		a lookup table. Only the beginning of entities are added to the table. Entities
		are evaluated on an entity- not token-level.
		
		Ex:		(e.g., Jason [B-PER], Camacho [I-PER]) * Camacho is ignored
	"""
	def __init__(self, sentences):
		self.entities = collections.defaultdict(str)
		self.default = 'O'
		self._create_lookup(sentences)

	def _create_lookup(self, sentences):
		"""
		Helper method to create a dictionary of single named entities.

		@params sentences: dataset
		@return: dictionary of entities
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

		* Additive smoothing (alpha=100) performs decently well.
		* A second order HMM would likely exceed this baseline.
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
		emissions.

		@params sentences: dataset
		@return: None
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
		Algorithm for decoding the most likely tag sequence by using dynamic programming. 

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
	A Maximum Entropy Markov Model (MEMM) a discriminative classifier powered by Logistic 
	Regression. Naively, the model can be understood as applying the classifier at each 
	time step using the current configuration as features.
	'''
	def __init__(self, sentences, regularization=1.0000, max_iters=1000, eta=0.01): # default = 0.00001
		self.label_encoder = LabelEncoder()
		self.feature_encoder = DictVectorizer()
		self.org_gazetteers = corpus.read_gazetteers('implementation/gazetteers/ned.list.ORG') 
		self.misc_gazetteers = corpus.read_gazetteers('implementation/gazetteers/ned.list.MISC')
		self.per_gazetteers = corpus.read_gazetteers('implementation/gazetteers/ned.list.PER')
		self.nlp = spacy.blank('en')
		self.embeddings = api.load("fasttext-wiki-news-subwords-300")
		# self.embeddings = api.load('word2vec-google-news-300')
		self.model = lr.SGDClassifier()
		self.model = SGDClassifier(penalty='l2', alpha=regularization, loss='log_loss', max_iter=max_iters, eta0=eta, learning_rate='constant', random_state=42)
		self.create_data(sentences)
		self.model.fit(self.X, self.y)

	def get_embeddings(self, word, emb_model):
		if word in emb_model.key_to_index:
			emb = emb_model[word]
		else:
			emb = np.zeros(emb_model.vector_size) 
		return emb

	def _extract_features(self, token, doc, i, prev_feats, s): #focus on new words analysis between embeddings
		"""
		Method for extracting features for each token in the NER dataset. The matrix is implicitly 
		constructed where each row is a token and contains a set of attributes that descrive it.

		@param token: word
		@param sentence: sentence used for extra features
		@param index: index for accessing these extra features
		@return: dictionary of features
		"""
		word = doc[i]
		prev_label, prev_pos, prev_word, prev_chunk = prev_feats

		features = {
				'isFirst': 1 if i == 0 else 0,#
				'numeric': token.word.isnumeric(),
				'dollar': '$' in token.word,
				'%': '%' in token.word,
				'word': token.word,
				'word_lower': word.lower_,
				'pos': token.pos,
				'chunk': token.chunk,
				'is_title': word.is_title,
				'prefix_': word.prefix_,
				'suffix_': word.suffix_,
				'hyphen': '-' in token.word,
				'+1_word': s[i+1].word.lower() if i < len(s) - 1 else '</s>', 
				'+1_pos': s[i+1].pos if i < len(s) - 1 else '</s>',
				'+2_word': s[i+2].word.lower() if i < len(s) - 2 else '</s>', 
				'+2_pos': s[i+2].pos if i < len(s) - 2 else '</s>', 
				'-1_label': prev_label,
				'-1_word': prev_word.lower(), 
				'-1_word_u': prev_word,
				'-1_pos': prev_pos,
				'-1_IN': 1 if prev_pos.startswith('IN') else 0,
				'-1_upper': 1 if s[i-1].word[0].isupper() and i > 0 else 0,
				'ORG': 'ORG' if token.word.lower() in self.org_gazetteers else 'NONE',
				'MISC': 'MISC' if token.word.lower() in self.misc_gazetteers else 'NONE',

		}
		features['POS'] = str(features['pos']) + str(features['-1_pos'])
		features['upper'] = str(features['is_title']) + str(features['-1_upper'])
		return features

	def create_data(self, sentences):
		"""
		Overloaded method that both creates the matrix and trains the MEMM classifier on 
		labeled sentences. The method also separates labels into a vector.

		@param sentences: dataset 
		@return: None
		"""
		matrix = []
		targets = []
		for sentence in sentences:
			text = ' '.join([token.word for token in sentence[1:-1]])
			doc = self.nlp(text)
			for i, token in enumerate(sentence[1:-1]):
				if i == 0:
					prev_label, prev_pos, prev_word = '<s>', '<s>', '<s>'
					prev_chunk = '<s>'
				else:
					inst = sentence[i-1]
					prev_label, prev_label, prev_word = inst.ne, inst.pos, inst.word
					prev_chunk = inst.chunk
				prev_feats = tuple((prev_label, prev_pos, prev_word, prev_chunk))

				text = ' '.join([token.word for token in sentence[1:-1]])
				doc = self.nlp(text)
				features = self._extract_features(token, doc, i, prev_feats, sentence)

				matrix.append(features)
				targets.append(token.ne)
		assert len(matrix) == len(targets)

		self.feature_encoder.fit(matrix)
		self.label_encoder.fit(targets)
		self.X = self.feature_encoder.transform(matrix)
		self.y = self.label_encoder.transform(targets) 

		all_embeddings = []
		for sentence in sentences:
			for i, token in enumerate(sentence[1:-1]):
				if i == 0:
					prev_embedding = csr_matrix(self.get_embeddings('<s>', self.embeddings))
				else: 
					prev_embedding = csr_matrix(self.get_embeddings(sentence[i-1].word, self.embeddings))
				curr_embeddings = csr_matrix(self.get_embeddings(token.word, self.embeddings))
				sparse_embeddings = hstack([prev_embedding, curr_embeddings]) #concatenation
				# sparse_embeddings = (prev_embedding.todense() + curr_embeddings.todense()) #addition /2
				sparse_embeddings = csr_matrix(sparse_embeddings)
				all_embeddings.append(sparse_embeddings) 

		embedding_matrix = vstack(all_embeddings)
		self.X = hstack([self.X, embedding_matrix])

	def predict(self, sentence):
		'''
		Simple method to decode the predicted sequence given by the algorithm. Greedy 
		decoding is the default option since it is the quickest and performs best.

		@params sentence:  sentence
		@return ne_labels: tag sequence of ne labels
		'''
		ne_labels = self.greedy_sequence_decoding(sentence) # DEFAULT DECODING STRATEGY
		# ne_labels = self.viterbi(sentence)
		return ne_labels

	def greedy_sequence_decoding(self, sentence):
		'''
		Greedy sequence decoding chooses the maximized label from left to right. This decoding 
		performs a hard decision at each time step. Therefore, when using greedy decoding the
		MEMM is a weak nth order markov model

		@params sentence: string of words
		@return tag sequence: sequence of ne labels
		'''
		tag_sequence = []
		text = ' '.join([token.word for token in sentence[1:-1]])
		doc = self.nlp(text)

		for i, token in enumerate(sentence[1:-1]):
			if i == 0:
				prev_label, prev_pos, prev_word = '<s>', '<s>', '<s>'
				prev_chunk = '<s>'
			else:
				inst = sentence[i-1]
				prev_label, prev_pos, prev_word = inst.ne, inst.pos, inst.word
				prev_chunk = inst.chunk
			prev_feats = tuple((prev_label, prev_pos, prev_word, prev_chunk))
			feats = self._extract_features(token, doc, i, prev_feats, sentence)
			encoded_feats = self.feature_encoder.transform(feats)
			if i == 0:
				prev_embedding = csr_matrix(self.get_embeddings('<s>', self.embeddings))
			else:
				prev_embedding = csr_matrix(self.get_embeddings(sentence[i-1].word, self.embeddings))					
			curr_embeddings = csr_matrix(self.get_embeddings(token.word, self.embeddings))
			sparse_embeddings = hstack([prev_embedding, curr_embeddings]) # concatenation
			# sparse_embeddings = (prev_embedding.todense() + curr_embeddings.todense()) # addition /2
			sparse_embeddings = csr_matrix(sparse_embeddings)

			total_feats = hstack([encoded_feats, sparse_embeddings]) 
			encoded_prediction = self.model.predict(total_feats)
			decoded_prediction = self.label_encoder.inverse_transform(encoded_prediction)[0]
			tag_sequence.append(decoded_prediction)

		return tag_sequence

	def viterbi(self, s): 
		s = s[1:-1]
		classes = self.label_encoder.classes_
		grid = np.zeros([len(classes), len(s)])
		best = np.zeros([len(classes), len(s)], dtype=int)
		text = ' '.join([token.word for token in s])
		doc = self.nlp(text)

		for i, label in enumerate(classes): 
			instance = s[0]
			prev_feats = ("<s>", "<s>", "<s>", "<s>") 

			feats = self._extract_features(instance, doc, 0, prev_feats, s) 

			feature_vector = self.feature_encoder.transform(feats)

			sparse_prev = csr_matrix(self.get_embeddings('<s>', self.embeddings))
			sparse_curr = csr_matrix(self.get_embeddings(instance.word, self.embeddings))
			sparse_emb = hstack([sparse_prev, sparse_curr])

			feature_vector = hstack([feature_vector, sparse_emb])

			log_probs = self.model.predict_log_proba(feature_vector)
			
			grid[i, 0] = log_probs.flatten()[i]

		for wpos in range(1, len(s)):
			instance = s[wpos]
			for i, label in enumerate(classes):
				val = []
				for j, curr_label in enumerate(classes):
					prev_inst = s[wpos-1]
					prev_feats = tuple((curr_label, prev_inst.pos, prev_inst.word, prev_inst.chunk)) #add embedding feature

					feats = self._extract_features(instance, doc, wpos, prev_feats, s)
					feature_vector = self.feature_encoder.transform(feats)

					sparse_prev = csr_matrix(self.get_embeddings(prev_inst.word, self.embeddings))
					sparse_curr = csr_matrix(self.get_embeddings(instance.word, self.embeddings))
					sparse_emb = hstack([sparse_prev, sparse_curr])

					feature_vector = hstack([feature_vector, sparse_emb])

					log_probs = self.model.predict_log_proba(feature_vector)

					curval = grid[j, wpos-1] + log_probs.flatten()[i]
					val.append(curval)
				grid[i][wpos] = max(val) 
				best[i][wpos] = np.argmax(val)

		last_col = grid.argmax(axis=0)[-1]
		res = [last_col]
		i = len(s)-1
		while i >= 1:
			res.append(best[res[-1], i])
			i -= 1
		res.reverse()

		res = self.label_encoder.inverse_transform(res)
		return res 

	def label(self, sentences):

		for k, sentence in enumerate(sentences):
			labels = self.predict(sentence) 

			for i, token in enumerate(sentence[1:-1]):
				token.ne = labels[i]
			print(f"{k} sentence DONE!")

		# Hack to catch mistakes from classifier output of label sequence
		# greedy decoding but what about viterbi?
		for k, sentence in enumerate(sentences):
			sentence = sentence[1:-1]
			for i, token in enumerate(sentence):
				if i != 0:
					prev_ne = sentence[i-1].ne
					curr_ne = token.ne 
					if prev_ne[0] == 'B' and prev_ne[-3:] != curr_ne[-3:] and curr_ne[0] == 'I':
						if prev_ne.endswith('LOC'): 
							token.ne = 'I-LOC'
							# token.ne = 'O'
						elif prev_ne.endswith('PER'):
							token.ne = 'I-PER'
							# token.ne = 'O'
						elif prev_ne.endswith('MISC'):
							token.ne = 'I-MISC'
							# token.ne = 'O'
						elif prev_ne.endswith('ORG'):
							token.ne = 'I-ORG'
							# token.ne = 'O'

			print(f"{k} sentence CLEANED!") 
