import argparse
import pdb

class Token:
	def __init__(self, word, pos, chunk, ne):
		self.word = word
		self.pos = pos
		self.chunk = chunk
		self.ne = ne

	def __str__(self):
		return f"{self.word} {self.pos} {self.chunk} {self.ne}"

class DeuToken:
	def __init__(self, word, lemma, pos, chunk, ne):
		self.word = word
		self.lemma = lemma
		self.pos = pos
		self.chunk = chunk
		self.ne = ne

class EspToken:
	def __init__(self, word, ne):
		self.word = word
		self.ne = ne

	def __str__(self):
		return f"{self.word} {self.lemma} {self.pos} {self.chunk} {self.ne}"

def read_conll_data(file, encoding='utf-8', lang='eng', padding=True, test=False):
	""" Reads a BIO data."""
	collection = []
	sentence = []
	
	with open(file, 'r', encoding=encoding, errors='ignore') as f:
		for line in f:
			line = line.strip()
			if not line.startswith('-DOCSTART-') and line:
				# padding beg here
				if lang == 'eng':
					word, pos, chunk, ne = line.split()
					if test: ne = 'UNK'
					sentence.append(Token(word, pos, chunk, ne))

				elif lang == 'esp':
					word, ne = line.split()
					if test: 
						ne = 'UNK'
					sentence.append(EspToken(word, ne))

			else:
				if sentence:
					collection.append(sentence)
					sentence = []
	if sentence:
		collection.append(sentence)

	if padding:
		add_padding(collection)

	return collection

def add_padding(sentences):
	for sentence in sentences:
		sentence.insert(0, Token('<s>','<s>','<s>', '<s>'))
		sentence.append(Token('</s>','</s>','</s>', '</s>'))

# python3 implementation/corpus.py corpora/train/eng/eng.train

