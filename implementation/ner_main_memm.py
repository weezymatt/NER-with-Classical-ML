import ner_system
import argparse
import corpus 
import utils
import numpy as np
import time
import pdb



# python3 implementation/ner_main_memm.py corpora/train/eng/eng.train corpora/val/eng/eng.testa
# python3 implementation/ner_main_memm.py corpora/train/eng/eng.train corpora/test/eng/eng.testb
def main(): 
	'''
	Testing for the MEMM system only on the English corpora. Grid search was used for 
	hyperparameter tuning and resulted in the [best] parameters. The default decoding
	strategy is greedy search.
	'''
	parser = argparse.ArgumentParser(description="Read the corpus.")
	parser.add_argument('ENG_PATH_TR',
						help="Path to training file with NE annotations.")
	parser.add_argument('ENG_PATH_TE',
						help="Path to validation/testing file with NE annotations.")
	args = parser.parse_args()
	tr_sents = corpus.read_conll_data(args.ENG_PATH_TR, encoding='utf-8', lang='eng')
	te_gold = corpus.read_conll_data(args.ENG_PATH_TE, encoding='utf-8', lang='eng')
	te_pred = corpus.read_conll_data(args.ENG_PATH_TE, test=True, encoding='utf-8', lang='eng')

	# Tuning per grid search
	lambda_ = 1e-06  
	lr = 0.1
	epochs = 15

	start_time = time.time()
	ner_sys = ner_system.MEMM
	ner_sys = ner_sys(tr_sents, regularization=lambda_, max_iters=epochs, eta=lr)

	ner_sys.label(te_pred)
	end_time = time.time()

	acc, f1, report = utils.calculate_f1measure(te_gold, te_pred)
	print(str(ner_sys))
	print(f"File: {args.ENG_PATH_TE}")
	print(f"Accuracy: {acc}")
	print(f'F1-score: {f1}')
	print(f"Report: \n{report}")
	print(f"Total time: {end_time - start_time} seconds")

if __name__ == '__main__':
	main()
