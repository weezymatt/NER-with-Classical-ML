# Reports

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

### English Evaluation on Tuned Models

The table below shows the evaluation on tuned models. The `(Me)` caption indicates the model was built from scratch, and the `-sk` suffix indicates it was imported from scikit-learn. $\lambda$ is regularization, $\eta$ is the learning rate, and $\alpha$ is the smoothing value.

| Run Description         | Decoding | Test | LOC  | MISC | ORG  | PER  | Overall |
|-------------------------|----------|------|------|------|------|------|---------|
| MEMM-sk  | Greedy  | Val  | 89.0 | 79.0 | 75.0 | 89.0 | **84.0** |
| $\lambda$=0.1, $\eta$=0.1, epochs=15  |          | Test | 84.0 | 72.0 | 70.0 | 83.0 | **78.0** |
| (Me)MM | Greedy  | Val  | 90.0 | 79.0 | 74.0 | 88.0 | **84.0** |
|  $\lambda$=0.1, $\eta$=0.1, epochs=15 |          | Test | 84.0 | 70.0 | 69.0 | 83.0 | **78.0** |
| HM(Me)       | Viterbi | Val  | 86.0 | 82.0 | 71.0 | 76.0 | 79.0   |
|  $\alpha$=100   |          | Test | 80.0 | 72.0 | 62.0 | 57.0 | 69.0   |



## Results
