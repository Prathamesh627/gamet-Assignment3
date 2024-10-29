## GAMeT- Assignment 3

This repository contains the implementations of all the questions that are put forth in [Assignment 3](https://docs.google.com/document/d/1zfuJH2ZUQ8XcUlF5EMhPIfwP7Bn5OK4-sICZFdKYg-8/edit?usp=sharing). 

Folder Question_1 contains the implementation of question 1, basically building a **Next-word Predictor** using Multi-Layer Perceptron (MLP). It contains files such as:
- `app.py`: Contains the code of a Streamlit app that creates an interface to generate the next words by choosing the available drop-down options. It also contains the typical framework that would follow after a certain combination is chosen, to generate the output. It also directs to choice of specific model paths (`.pth` files), and mapping dictionaries (`.pkl` files) based on the selected combination.
- `model.py`: It contains the MLP model architecture used to implement the solution.
- `question_1_model_9_code.ipynb`: It provides a detailed view of training the MLP model based on certain hyperparameters. In this case, *model 9* is trained wherein **ReLU** is used as an activation function, `block_size=16`, `emb_dim=128`, and `hidden_size=1024`. After training the model is saved as `.pth` file and its integer-to-string and string-to-integer mappings are stored `.pkl` files. Then, some specific words are chosen from the vocabulary and are visualized using 2-dimensional t-SNE. This shows how that model relates the chosen words from the vocabulary.
