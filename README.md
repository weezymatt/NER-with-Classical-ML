# Revisiting CoNNL-2003 with Classical Machine Learning
## Table of Contents
- [Highlights](#highlights)
- [Directory](#directory)
- [Virtual Environment](#virtual-environment)
- [Data](#data)
- [Hyperparameters](#hyperparameters)
- [Main System](#main-system)
- [Improved System](#improved-system)

## Highlights
Contributors: Matthew A Hernandez
> **Note** Access to our paper [here](https://github.com/weezymatt/NER-with-Classical-Machine-Learning/blob/main/reports/INFO_521_report.pdf)

We present a brief survey on classical machine learning algorithms in the context of the CoNNL-2003 Shared Task. A named entity recognition (NER) system is built to recognize and classify objects in a body of text into predefined categories. We include a principled framework that motivates the use of machine learning by creating two baseline systems. Finally, the paper includes an analysis between generative and discriminative machine learning algorithms.

## Directory 

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

## Virtual Environment
The environment can be replicated with a virtual environment. Please follow the directions below to run the experiments from the paper.

```bash
$ git clone YOUR_REPO
$ cd NER-with-Classical-Machine-Learning
$ python3 -m venv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
```
## Data
We used both Spanish (CoNNL-2002) and English (CoNNL-2003) dataset. 

More information is in the ```\corpora``` subdirectory. 

## Hyperparameters
The hyperparameters were tuned with grid search to find the optimal values for regularization, epochs, and learning rate. 

<details>
  <summary>Click to expand!</summary>

  | Parameter | Value  |
|-----------|--------|
| $\lambda$ | 0.1    |
| $\eta$    | 0.1    |
| epochs    | 15     |
| $\alpha$  | 100    |

</details>

## Main System
The main NER system is located in ```implementation/ner_main.py``` and reports the results for the two baselines and Hidden Markov Model on the English/Spanish dataset.

```bash
python implementation/ner_main.py corpora/train/eng/eng.train corpora/val/eng/eng.testa corpora/test/eng/eng.testb corpora/train/esp/esp.train corpora/test/esp/esp.testb
```
## Improved System
The improved NER system is located in ```implementation/ner_main_memm.py``` and reports the results of the Maximum-entropy model on the English dataset. The default ME model is the SGDClassifier from scikit-learn and expected run time is around 3 minutes.

> We do not advise the user to switch the model to the (Me)MM because training time is significant. 

1. Validation set
```bash
python implementation/ner_main_memm.py corpora/train/eng/eng.train corpora/val/eng/eng.testa
```

2. Test set
```bash
python implementation/ner_main_memm.py corpora/train/eng/eng.train corpora/test/eng/eng.testb
```


