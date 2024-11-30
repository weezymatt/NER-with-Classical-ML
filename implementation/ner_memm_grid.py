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

	#GRID SEARCH — Improve by nested for-loops instead.
	hyperparams_reg = [1e-03, 1e-04, 1e-05, 1e-06, 1e-07]
	hyperparams_lr = [1e-01, 1e-02, 1e-03, 1e-04, 1e-05]
	hyperparams_epochs = [1, 5, 10, 15, 20]

	max_f1 = float('-inf')
	# grid_1 = np.zeros([len(hyperparams_reg), len(hyperparams_lr)])
	# grid_2 = np.zeros([len(hyperparams_reg), len(hyperparams_epochs)])
	grid_3 = np.zeros([len(hyperparams_lr), len(hyperparams_epochs)])

	# def_epochs = 15
	# for idx1, lambda_ in enumerate(hyperparams_reg):
	# 	for idx2, lr in enumerate(hyperparams_lr):
	# 		print(f"New system initialized! {(lambda_, lr)}")
	# 		ner_sys = ner_system.MEMM
	# 		ner_sys = ner_sys(tr_sents, regularization=lambda_, max_iters=def_epochs, eta=lr)
	# 		ner_sys.label(te_pred)
	# 		acc, f1, report = utils.calculate_f1measure(te_gold, te_pred)
	# 		grid_1[idx1, idx2] = f1
	# 		if f1 > max_f1:
	# 			max_f1 = f1
	# 			best_combo = tuple((lambda_, lr, report))

	# def_lr = 0.1
	# for idx1, lambda_ in enumerate(hyperparams_reg):
	# 	for idx2, epochs in enumerate(hyperparams_epochs):
	# 		print(f"New system initialized! {(lambda_, epochs)}")
	# 		ner_sys = ner_system.MEMM
	# 		ner_sys = ner_sys(tr_sents, regularization=lambda_, max_iters=epochs, eta=def_lr)
	# 		ner_sys.label(te_pred)
	# 		acc, f1, report = utils.calculate_f1measure(te_gold, te_pred)
	# 		grid_2[idx1, idx2] = f1
	# 		if f1 > max_f1:
	# 			max_f1 = f1
	# 			best_combo = tuple((lambda_, epochs, report))

	def_lambda = 1e-07
	for idx1, epochs in enumerate(hyperparams_epochs):
		for idx2, lr in enumerate(hyperparams_lr):
			print(f"New system initialized! {(epochs, lr)}")
			ner_sys = ner_system.MEMM
			ner_sys = ner_sys(tr_sents, regularization=def_lambda, max_iters=epochs, eta=lr)
			ner_sys.label(te_pred)
			acc, f1, report = utils.calculate_f1measure(te_gold, te_pred)
			grid_3[idx1, idx2] = f1
			if f1 > max_f1:
				max_f1 = f1
				best_combo = tuple((epochs, lr, report))

	print(f"Highest F1-measure: {max_f1}")
	print(f"Best combo: {best_combo[:-1]}")
	print(f"Report: {best_combo[-1]}")
	print()
	print(grid_3) # for paper