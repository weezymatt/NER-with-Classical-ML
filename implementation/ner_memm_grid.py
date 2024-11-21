import ner_system
import argparse
import corpus 
import utils
import numpy as np
import pdb

if __name__ == '__main__':
	'''
	Script for testing the hyperparameters (i.e., epochs, regularization, and eta)
	for the MEMM system.
	'''
	parser = argparse.ArgumentParser(description="Read the corpus.")
	parser.add_argument('ENG_PATH_TR',
						help="Path to training file with NE annotations.")
	parser.add_argument('ENG_PATH_TE',
						help="Path to testing file with NE annotations.")
	args = parser.parse_args()
	tr_sents = corpus.read_conll_data(args.ENG_PATH_TR, encoding='utf-8', lang='eng')

	te_gold = corpus.read_conll_data(args.ENG_PATH_TE, encoding='utf-8', lang='eng')
	te_pred = corpus.read_conll_data(args.ENG_PATH_TE, test=True, encoding='utf-8', lang='eng')

	#GRID SEARCH
	hyperparams_epochs = [1000, 2000, 3000, 4000, 5000, 6000] # best 1000
	hyperparams_reg = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001] 
	hyperparams_lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]

	max_f1 = float('-inf')
	grid = np.zeros([len(hyperparams_reg), len(hyperparams_lr)])

	for idx1, lambda_ in enumerate(hyperparams_reg):
		for idx2, lr in enumerate(hyperparams_lr):
			print(f"New system initialized! {(lambda_, lr)}")
			ner_sys = ner_system.MEMM
			ner_sys = ner_sys(tr_sents, regularization=lambda_, max_iters=epochs, eta=lr)
			ner_sys.label(te_pred)
			acc, f1, report = utils.calculate_f1measure(te_gold, te_pred)
			grid[idx1, idx2] = f1
			if f1 > max_f1:
				max_f1 = f1
				best_combo = tuple((lambda_, lr, report))

	print(f"Highest F1-measure: {max_f1}")
	print(f"Best combo: {best_combo[:-1]}")
	print(f"Report: {best_combo[-1]}")
	print()
	print(grid) # for paper