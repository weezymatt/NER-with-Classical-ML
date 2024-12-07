# Revisiting CoNNL-2003 with Classical Machine Learning
Contributors: Matthew A Hernandez
> **Note** Access to our paper [here](https://github.com/weezymatt/NER-with-Classical-Machine-Learning/blob/main/reports/INFO_521_report.pdf)


Last updated November 21st, 2024

> **Note** The scope of this project involves the following points: establish multiple baselines to support machine learning methods and provide a survey of classical machine learning to motivate more complex modelling (e.g., CRFs and LSTMs) for NER.

This project was created for the purpose of applying techniques in Machine Learning built from scratch to develop a Named Entity Recognition system. The main comparision is between a first-order Hidden Markov Model and a Maximum Entropy (ME) Markov Model.

> **Note 2** The Hidden Markov Model, Maximum Entropy, and all decoding strategies are built from scratch. The analysis in the paper compares the models built from scratch with the scikit-learn's version of Logistic Regression.

## Directory Layout

```bash
.
├── corpora/                # Contains datasets used for training and evaluation
│   ├── test/               
│   ├── train/          
│   └── val/                
├── implementation/         # Source code for the project
│   ├── gazetteers/         # Folder for text files of named entities
│   ├── models/             # SGDClassifier (scratch)
│   ├── utils/              # Utilities for reporting accuracy and exact-entity eval
│   ├── corpus.py           # Script to read the CoNLL data
│   ├── ner_main.py         # The full NER system 
│   ├── ner_main_memm.py    # Script to train MEMM and report results
│   ├── ner_memm_grid.py    # Script to use grid search for tuning
│   └── ner_system.py       # Script to train HMM and report results
├── reports/                # Text files of various reports
│   └── benchmarks/         
│   └── grid-search/     
├── README.md               
├── requirements.txt        
```


## Table of Contents
- [Objective](#objective)
- [Virtual Environment](#virtual-environment)
- [Structure](#structure)
- [Baselines](#baselines)
- [Main System](#main-system)
- [Improved System](#improved-system)
- [Results](#results)
- [Citations](#citations)

## Objective
This repository presents a brief survey on classical machine learning algorithms in the context of the CoNNL-2003 Shared Task. A Named Entity Recognition (NER) system is built to recognize and classify objects in a body of text into predefined categories. The paper includes a principled framework that motivates the use of machine learning. Finally, the paper includes an analysis between generative and discriminative machine learning algorithms with various decoding methods for inference.

## Virtual Environment
The environment can be replicated with a virtual environment. Please follow the directions below to run the experiments from the paper.

```bash
$ git clone YOUR_REPO
$ cd NER-with-Classical-Machine-Learning
$ python3 -m venv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
```

## Main System
The main NER system is located in ```implementation/ner_main.py``` and reports the results for the two baselines and Hidden Markov Model on the English/Spanish dataset.

```bash
python3 implementation/ner_main.py corpora/train/eng/eng.train corpora/val/eng/eng.testa corpora/test/eng/eng.testb corpora/train/esp/esp.train corpora/test/esp/esp.testb
```
## Improved System
The improved NER system is located in ```implementation/ner_main_memm.py``` and reports the results of the Maximum-entropy model on the English dataset. The default ME model is the SGDClassifier from scikit-learn and expected run time is around 3 minutes.

> We do not advise the user to switch the model to the (Me)MM because training time is significant. 

1. Validation set
```bash
python3 implementation/ner_main.py corpora/train/eng/eng.train corpora/val/eng/eng.testa
```

2. Test set
```bash
python3 implementation/ner_main.py corpora/train/eng/eng.train corpora/test/eng/eng.testb
```

## Results
The `(Me)` caption indicates the model is built from scratch. Models with the `-sk` suffix are imported from Scikit-learn.

#### English Evaluation on Tuned Models

| Run Description     | Decoding | Test/Val | LOC  | MISC | ORG  | PER  | Overall  |
|---------------------|----------|----------|------|------|------|------|----------|
| 1. MEMM-sk          | Greedy   | Val      | 91.  | 80.  | 76.  | 90.  | **84.53** |
|                     |          | Test     | 84.  | 72.  | 73.  | 85.  | **77.81** |
| 2. (Me)MM           | Greedy   | Val      | 89.  | 78.  | 75.  | 90.  | 83.11    |
|                     |          | Test     | 82.  | 70.  | 69.  | 85.  | 75.64    |
| 3. HM(Me)           | Viterbi  | Val      | 86.  | 82.  | 71.  | 76.  | 79.01    |
|                     |          | Test     | 80.  | 72.  | 62.  | 57.  | 68.45    |

#### Spanish Evaluation on Tuned Model

| Run Description | Decoding | Test | LOC  | MISC | ORG  | PER  | Overall  |
|-----------------|----------|------|------|------|------|------|----------|
| 1. HM(Me)       | Greedy   | Test | 71.  | 37.  | 72.  | 69.  | **67.5** |

## Citations
[1] Erik F. Tjong Kim Sang and Fien De Meulder. Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. In: Proceedings of CoNLL-2003. Edmonton, Canada, 2003.
