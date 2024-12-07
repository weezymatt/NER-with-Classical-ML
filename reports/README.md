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
| `AlwaysNonEntity`   | 83.2    | 0.0    | 82.2     | 0.0     |
| `SingleEntity`      | 86.2    | 40.0   | 84.8     | **35.0** |

**Table 2: Results on Spanish test set**

| Model               | Test Acc | Test F1 |
|---------------------|----------|---------|
| `AlwaysNonEntity`   | 88.0     | 0.0     |
| `SingleEntity`      | 74.4     | **17.0** |

## Experiments

The table below shows the evaluation on tuned models. The `(Me)` caption indicates the model was built from scratch, and the `-sk` suffix indicates it was imported from scikit-learn. $\lambda$ is regularization, $\eta$ is the learning rate, and $\alpha$ is the smoothing value. 

The table below shows the tuned models with $\lambda$=0.1, $\eta$=0.1, epochs=15, and $\alpha$=100.

**Table 3: English evaluation on tuned models**

| Run Description         | Decoding | Test | LOC  | MISC | ORG  | PER  | Overall |
|-------------------------|----------|------|------|------|------|------|---------|
| MEMM-sk  | Greedy  | Val  | 89.0 | 79.0 | 75.0 | 89.0 | **84.0** |
|    |          | Test | 84.0 | 72.0 | 70.0 | 83.0 | **78.0** |
| (Me)MM | Greedy  | Val  | 90.0 | 79.0 | 74.0 | 88.0 | **84.0** |
|    |          | Test | 84.0 | 70.0 | 69.0 | 83.0 | **78.0** |
| HM(Me)       | Viterbi | Val  | 86.0 | 82.0 | 71.0 | 76.0 | 79.0   |
|     |          | Test | 80.0 | 72.0 | 62.0 | 57.0 | 69.0   |


The table below shows the Spanish evaluation on the tuned model with $\alpha=100$. The `(Me)` caption indicates the model was built from scratch.

**Table 4: Spanish evaluation on tuned models.**

| Run Description | Decoding | Test | LOC  | MISC | ORG  | PER  | Overall |
|-----------------|----------|------|------|------|------|------|---------|
| 1. HM(Me)  | Greedy  | Test | 71.0 | 37.0 | 72.0 | 69.0 | **68.0** |

## Citations
[1] Erik F. Tjong Kim Sang and Fien De Meulder. Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. In: Proceedings of CoNLL-2003. Edmonton, Canada, 2003.
