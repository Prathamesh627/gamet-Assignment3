## GAMeT- Assignment 3

This repository contains the implementations of all the questions that are put forth in [Assignment 3](https://docs.google.com/document/d/1zfuJH2ZUQ8XcUlF5EMhPIfwP7Bn5OK4-sICZFdKYg-8/edit?usp=sharing). 

Folder **Question_1** contains the implementation of question 1, basically building a **Next-word Predictor** using Multi-Layer Perceptron (MLP). It contains files such as:
- `app.py`: Contains the code of a Streamlit app that creates an interface to generate the next words by choosing the available drop-down options. It also contains the typical framework that would follow after a certain combination is chosen, to generate the output. It also directs to choice of specific model paths (`.pth` files), and mapping dictionaries (`.pkl` files) based on the selected combination.
- `model.py`: It contains the MLP model architecture used to implement the solution.
- `question_1_model_9_code.ipynb`: It provides a detailed view of training the MLP model based on certain hyperparameters. In this case, *model 9* is trained wherein **tanh** is used as an activation function, `block_size=16`, `emb_dim=128`, and `hidden_size=1024`. After training the model is saved as `.pth` file and its integer-to-string and string-to-integer mappings are stored `.pkl` files. Then, some specific words are chosen from the vocabulary and are visualized using 2-dimensional t-SNE. This shows how that model relates the chosen words from the vocabulary.
- `requirements.txt`: It lists the requirements of modules required to implement the `app.py`.
- `sherlock.txt`: It is a dataset on which the vocabulary has been built and the MLP model has been trained. Source: [Project Gutenberg](https://www.gutenberg.org/files/1661/1661-0.txt)

**question2.ipynb** implements question 2 in which the XOR dataset is generated and learned with the models of MLP, MLP w/ L1 regularization, MLP w/ L2 regularization, and logistic regression.

**quesion3.ipynb** implements question 3 wherein multiple models like MLP, Random Forest, and Logistic Regression are trained and tested on the MNIST dataset. F1-score, confusion matrices, and embedding visualizations are used for their comparative analysis. Further, the MLP is used to predict from the Fashion-MNIST dataset and visualize the embeddings.
