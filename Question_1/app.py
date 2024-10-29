import torch
import streamlit as st
import re
import torch.nn.functional as F
import pickle

from model import NextWord

# Select boxes for each parameter
st.text('Hello, welcome to playground of ML Assignment-3!')
embedding_choice = st.selectbox('Choose embedding size:', [64, 128])
hidden_choice = st.selectbox('Choose hidden layer size:', [512, 1024])
block_size_choice = st.selectbox('Choose context size (block size):', [12, 16])
activation_choice = st.selectbox('Choose the activation function:', ['tanh','relu'])

# Set the model parameters based on user choices
vocab_size = 8690  
embedding_dim = embedding_choice
hidden_dim = hidden_choice  
block_size = block_size_choice
activation=activation_choice

# Initialize the model
model = NextWord(block_size, vocab_size, embedding_dim, hidden_dim, activation)


# Function to remove the "_orig_mod." prefix in the state dict keys
def remove_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

# Mapping combinations to file numbers
file_map = {
    (64, 512, 16, 'relu'): 1,
    (64, 1024, 16, 'relu'): 2,
    (128, 512, 16, 'relu'): 3,
    (128, 1024, 16, 'relu'): 4,
    (64, 512, 12, 'relu'): 5,
    (64, 1024, 12, 'relu'): 6,
    (128, 512, 12, 'relu'): 7,
    (128, 1024, 12, 'relu'): 8,
    (128, 1024, 16, 'tanh'): 9,
    (128, 1024, 12, 'tanh'): 10,
    (128, 512, 16, 'tanh'): 11,
    (128, 512, 12, 'tanh'): 12,
    (64, 512, 16, 'tanh'): 13,
    (64, 512, 12, 'tanh'): 14,
    (64, 1024, 16, 'tanh'): 15,
    (64, 1024, 12, 'tanh'): 16,
}

# Get the file number based on the user selection
file_number = file_map.get((embedding_dim, hidden_dim, block_size, activation))

# Construct the file names
model_filename = f"model_sherlock_{file_number}.pth"
stoi_filename = f"stoi_{file_number}.pkl"
itos_filename = f"itos_{file_number}.pkl"

# Load the model state_dict
try:
    state_dict = torch.load(model_filename, map_location=torch.device('cpu'))
    state_dict = remove_prefix_from_state_dict(state_dict)
    model.load_state_dict(state_dict)
except FileNotFoundError:
    st.error(f"Model file '{model_filename}' not found. Please upload the correct file.")

# Set the model to evaluation mode
model.eval()

# Load the word mappings (stoi and itos)
try:
    with open(stoi_filename, 'rb') as f:
        stoi = pickle.load(f)
    with open(itos_filename, 'rb') as f:
        itos = pickle.load(f)
except FileNotFoundError:
    st.error(f"Mapping files '{stoi_filename}' or '{itos_filename}' not found. Please upload the correct files.")

# Load and preprocess the text data
with open('sherlock.txt', 'r', encoding='utf-8') as f:
    data = f.read()

data = data.replace('\n', '  ')
data = re.sub(r'([a-zA-Z0-9])([\.])', r'\1 \2', data) 
data = re.sub('[^a-zA-Z0-9 \.]', '', data).lower()
words = data.split()
words = words[254:]  

# Text generation function
def generate_text(model, initial_text, stoi, itos, block_size, k=50):
    context = initial_text.split()[-block_size:]  
    context_ids = [stoi.get(word, 0) for word in context]  
    context_ids = [0] * (block_size - len(context_ids)) + context_ids  
    generated_text = initial_text

    for _ in range(k):
        x = torch.tensor(context_ids).view(1, -1)

        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()

        next_word = itos[ix]
        generated_text += ' ' + next_word

        context_ids = context_ids[1:] + [ix]

    return generated_text


input_text = st.text_input("Enter input text (don't give very short prompts):", placeholder="Enter something")

if input_text == "":
    st.write("This is a Sherlock Holmes book trained model, please type some text input.")
else:
    output_text = generate_text(model, input_text, stoi, itos, block_size)
    st.write(f"Predicted next word(s): {output_text}")

