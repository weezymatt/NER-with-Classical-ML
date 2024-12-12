from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
import corpus
import pdb

# https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
# https://learn.microsoft.com/en-us/azure/ai-services/language-service/custom-named-entity-recognition/concepts/evaluation-metrics

def calculate_accuracy(gold, predictions):
	"""
	A majority of tokens are non-entities, therefore accuracy & f1-score 
	at the token-level is not a good indication of performance for the model.

	Example:
		English accuracy: 82.53041886508022
		German accuracy: 90.04485686232985
	"""
	gold = list(gold)
	predictions = list(predictions)
	assert len(gold) == len(predictions)

	correct = []
	for gold_sent, pred_sent in zip(gold, predictions):
		for gold_tok, pred_tok in zip(gold_sent, pred_sent):
			correct.append(gold_tok.ne == pred_tok.ne)

	return (correct.count(True)/ float(len(correct))) * 100

def convert_format(sentences):
	list_form = []
	for sentence in sentences: 

		sentence = sentence[1:-1] # changed here
		labels = []
		for token in sentence:
			labels.append(token.ne)
		list_form.append(labels)
		assert len(labels) == len(sentence)
	assert len(list_form) == len(sentences)

	return list_form

def calculate_f1measure(gold, predictions):
	"""
	CoNLL definition:
		A named entity is correct only if it is an exact match 
		of the corresponding entity in the data file.

	Args: 
		Ex: [['O', 'O', 'O', 'B-LOC', 'O']]

		gold: gold labels as a list of lists
		pred: pred labels as a list of lists
	Returns:
		accuracy: accuracy 
		f1 score: exact-match (excludes 'O' label)
		report: detailed class report
	"""
	# corpus.add_padding(gold)
	y_true = convert_format(gold)
	y_pred = convert_format(predictions)

	accuracy = accuracy_score(y_true, y_pred)
	score = f1_score(y_true, y_pred)
	report = classification_report(y_true, y_pred, mode='strict', scheme=IOB2, zero_division=1, digits=4)
	# pdb.set_trace()
	return accuracy, score, report
