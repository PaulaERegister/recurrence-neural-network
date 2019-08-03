from IPython.core.interactiveshell import InteractiveShell
from IPython.display import HTML
import warnings
import pandas as pd
import numpy as np
from utils import get_data, generate_output, guess_human, seed_sequence, get_embeddings, find_closest

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.optimizers import Adam

from keras.utils import plot_model
InteractiveShell.ast_node_interactivity = 'all'

def read_file(file_name):
    data = pd.read_csv(file_name)
    print(data.head())
    training_dict, word_idx, idx_word, sequences = get_data(file_name, training_len=50)
    print(training_dict['X_train'][:2])
    print(training_dict['y_train'][:2])
    for i, sequence in enumerate(training_dict['X_train'][:2]):
        text = []
        for idx in sequence:
            text.append(idx_word[idx])

        print('Features: ' + ' '.join(text) + '\n')
        print('Label: ' + idx_word[np.argmax(training_dict['y_train'][i])] + '\n')
    create_model(word_idx)
    return training_dict

def create_model(word_idx):
    model = Sequential()

    model.add(Embedding(input_dim=len(word_idx) + 1, output_dim=100, weights=None, trainable=True))
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(word_idx) + 1, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

def train_model(training_dict):
    model = load_model('./models/train-embeddings-rnn.h5')
    h = model.fit(training_dict['X_train'], training_dict['y_train'], epochs = 5, batch_size = 2048,
          validation_data = (training_dict['X_valid'], training_dict['y_valid']),
          verbose = 1)
    model = load_model('../models/train-embeddings-rnn.h5')
    print('Model Performance: Log Loss and Accuracy on training data')
    model.evaluate(training_dict['X_train'], training_dict['y_train'], batch_size=2048)

    print('\nModel Performance: Log Loss and Accuracy on validation data')
    model.evaluate(training_dict['X_valid'], training_dict['y_valid'], batch_size=2048)

def main():
    training_dict = read_file('./data/neural_network_patent_query.csv')
    train_model(training_dict)

main()

