# Emotion Analysis with BERT and DistilBert
This repository contains code for fine-tuning pre-trained BERT and DistilBert models for emotion analysis on a Twitter dataset.

## Overview
The goal of this project is to use Twitter data to train models to recognize emotions in text. The code demonstrates how to clean and preprocess the data, tokenize it, and fine-tune pre-trained models on it. Additionally, there's code for hyperparameter tuning to optimize the performance of the models.

## Dataset
The dataset used in this project is the "Twitter-Sentiment-Analysis" dataset, which can be found [here](https://www.kaggle.com/datasets/ankitkumar2635/sentiment-and-emotions-of-tweets).

## Pre-trained Models
- [DistilBert](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base): A distilled version of BERT, optimized for faster training and inference without significant loss in performance.
- [BERT](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion): A transformer-based model trained for various NLP tasks.

## Usage
Run the provided Jupyter notebook (Emotion_Analysis.ipynb) to:

- Load and preprocess the data.
- Tokenize the data.
- Fine-tune DistilBert and BERT models.
- Visualize the training and validation accuracy and loss.
- View the classification report and confusion matrix for the models.
- Perform hyperparameter tuning.

## Results
The code includes plots for training and validation accuracy and loss to help visualize the model's performance. Additionally, there's a classification report and confusion matrix to evaluate the model's predictions on the validation set.

## Hyperparameter Tuning
The code uses Keras Tuner to perform hyperparameter tuning for the BERT-based model. This helps in finding the optimal learning rate and dropout rate for the model.
