import ner_system
import argparse
import corpus 
import utils
import pdb


# TESTING FOR THE MAIN SYSTEM EXCLUDING THE MEMM 
# python3 implementation/ner_main.py corpora/train/eng/eng.train corpora/val/eng/eng.testa corpora/test/eng/eng.testb corpora/train/esp/esp.train corpora/test/esp/esp.testb
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Read the corpus.")
	parser.add_argument('ENG_PATH_TR',
						help="Path to training file with NE annotations.")
	parser.add_argument('ENG_PATH_VA',
						help="Path to validation file with NE annotations.")
	parser.add_argument('ENG_PATH_TE',
						help="Path to testing file with NE annotations.")

	parser.add_argument('ESP_PATH_TR',
						help="Path to Spanish training file with NE annotations.") 
	parser.add_argument('ESP_PATH_TE',
						help="Path to Spanish testing file with NE annotations.")

	args = parser.parse_args()
	
	tr_sents = corpus.read_conll_data(args.ENG_PATH_TR, encoding='utf-8', lang='eng')
	esp_tr_sents = corpus.read_conll_data(args.ESP_PATH_TR, encoding='latin-1', lang='esp') 

	for ner_sys in [ner_system.AlwaysNonEntity, ner_system.SingleEntity, ner_system.FirstOrderHMM]:
		ner_sys1, ner_sys2 = ner_sys(tr_sents), ner_sys(esp_tr_sents) 
		for te_path in [args.ENG_PATH_VA, args.ENG_PATH_TE]:
			te_gold = corpus.read_conll_data(te_path, encoding='utf-8', lang='eng')
			te_pred = corpus.read_conll_data(te_path, test=True, encoding='utf-8', lang='eng')

			ner_sys1.label(te_pred)

			acc, f1, report = utils.calculate_f1measure(te_gold, te_pred)
			print(str(ner_sys))
			print(f"File: {te_path}")
			print(f"Accuracy: {acc}")
			print(f'F1-score: {f1}')
			print(f"Report: \n{report}")

		for esp_te_path in [args.ESP_PATH_TE]:
			te_gold2 = corpus.read_conll_data(esp_te_path, encoding='latin-1', lang='esp')
			te_pred2 = corpus.read_conll_data(esp_te_path, test=True, encoding='latin-1', lang='esp')
			ner_sys2.label(te_pred2)

			acc, f1, report = utils.calculate_f1measure(te_gold2, te_pred2)
			print(str(ner_sys))
			print(f"File {esp_te_path}")
			print(f"Accuracy: {acc}")
			print(f'F1-score: {f1}')
			print(f"Report: \n{report}")
