import ner_system
import argparse
import corpus 
import utils
import pdb

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Read the corpus.")
	parser.add_argument('PATH_TR',
						help="Path to training file with NER annotations.")
	parser.add_argument('PATH_TE',
						help="Path to testing file with NER annotations.")
	args = parser.parse_args()
	
	tr_sents = corpus.read_conll_data(args.PATH_TR, encoding='utf-8', lang='eng')

	# for ner_sys in [ner_system.AlwaysNonEntity, ner_system.SingleEntity, ner_system.FirstOrderHMM, ner_system.SecondOrderHMM]:
	# for ner_sys in [ner_system.SecondOrderHMM]:
	for ner_sys in [ner_system.MEMM]:
		ner_sys = ner_sys(tr_sents)
		for te_path in [args.PATH_TE]:
			te_gold = corpus.read_conll_data(te_path, encoding='utf-8', lang='eng')

			te_pred = corpus.read_conll_data(te_path, test=True, encoding='utf-8', lang='eng')
			ner_sys.label(te_pred)

			acc, f1, report = utils.calculate_f1measure(te_gold, te_pred)
			print(str(ner_sys))
			print(f"Accuracy: {acc}")
			print(f'F1-score: {f1}')
			print(f"Report: \n{report}")
