# Toxic Comment Classification Project

This project aims to classify and analyze toxic comments from online platforms using various machine learning models. The goal is to identify different kinds of toxicity like threats, insults, and identity-based hate speech.

## Overview

The repository contains a Jupyter notebook with the entire machine learning pipeline from preprocessing to model evaluation, as well as a `utils.py` file containing all the necessary utility functions. The models are evaluated based on their F1 scores across multiple categories of toxicity to gauge their effectiveness in distinguishing between different kinds of toxic comments.

## Models

The following models are implemented and evaluated:

- Logistic Regression (LogReg)
- Naive Bayes (NB)
- Recurrent Neural Networks (RNN) with numerical input (rnn3_num and rnn4_num_cust_loss)
- Basic RNN models (rnn1 and rnn2)

## Key Findings

- Logistic Regression exhibited the best overall performance with the highest weighted average F1 score.
- RNN models incorporating numerical inputs and custom loss functions demonstrated improved performance over simpler RNN architectures, particularly in the 'severe_toxic' and 'identity_hate' categories.
- There remains a challenge in accurately detecting 'threat' across all models, indicating a potential area for future research and model improvement.

## Utility Functions (`utils.py`)

The `utils.py` file includes functions to support the machine learning workflow:

- Custom loss functions
- Model creation functions
- Preprocessing utilities
- Functions to calculate evaluation metrics
