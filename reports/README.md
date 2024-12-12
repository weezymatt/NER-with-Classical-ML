# Reports
We adopt micro-F1 to better represent class imbalances as stated by Tjong Kim Sang and
De Meulder [1]. The *overall* metric is simply the micro-averaged F1 measure.
## Baselines
Two rule-based systems were computed for the English and Spanish corpora. The
AlwaysNonEntity is a naive baseline that labels all tokens as ’O’ or as non-entities, see
Table 1. The less naive baseline SingleEntity is based on a lookup table where only the
beginning (B-) of entities are added, see Table 2.


**Table 1: Results on English dev and test set (Accuracy and Micro F1).**

| Model               | Dev Acc | Dev F1 | Test Acc | Test F1 |
|---------------------|---------|--------|----------|---------|
| `AlwaysNonEntity`   | 83.3    | 0.0    | 82.5     | 0.0     |
| `SingleEntity`      | 86.5    | 40.0   | 84.9     | **35.4** |

**Table 2: Results on Spanish test set**

| Model               | Test Acc | Test F1 |
|---------------------|----------|---------|
| `AlwaysNonEntity`   | 88.0     | 0.0     |
| `SingleEntity`      | 74.5     | **16.9** |

## Experiments

The table below shows the evaluation on tuned models. The `(Me)` caption indicates the model was built from scratch, and the `-sk` suffix indicates it was imported from scikit-learn. $\lambda$ is regularization, $\eta$ is the learning rate, and $\alpha$ is the smoothing value. 

The table below shows the tuned models with $\lambda$=0.1, $\eta$=0.1, epochs=15, and $\alpha$=100.

**Table 3: English evaluation on tuned models**

| Run Description         | Decoding | Test | LOC  | MISC | ORG  | PER  | Overall |
|-------------------------|----------|------|------|------|------|------|---------|
| MEMM-sk  | Greedy  | Val  | 89.3 | 79.1 | 75.2 | 89.3 | **84.5** |
|    |          | Test | 84.0 | 72.2 | 70.5 | 83.5 | **78.3** |
| (Me)MM | Greedy  | Val  | 89.7 | 79.4 | 74.5 | 88.0 | 84.2 |
|    |          | Test | 84.0 | 70.4 | 68.8 | 82.7 | 77.6 |
| HM(Me)       | Viterbi | Val  | 85.7 | 82.5 | 71.4 | 76.1 | 79.1   |
|     |          | Test | 80.3 | 71.6 | 62.4 | 57.3 | 68.2   |


The table below shows the Spanish evaluation on the tuned model with $\alpha=100$. The `(Me)` caption indicates the model was built from scratch.

**Table 4: Spanish evaluation on tuned models.**

| Run Description | Decoding | Test | LOC  | MISC | ORG  | PER  | Overall |
|-----------------|----------|------|------|------|------|------|---------|
| 1. HM(Me)  | Greedy  | Test | 71.1 | 37.3 | 72.5 | 69.9 | **67.9** |

## Citations
[1] Erik F. Tjong Kim Sang and Fien De Meulder. Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. In: Proceedings of CoNLL-2003. Edmonton, Canada, 2003.
