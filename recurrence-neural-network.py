from IPython.core.interactiveshell import InteractiveShell
from IPython.display import HTML
import warnings
import pandas as pd
import numpy as np
from utils import get_data, generate_output, guess_human, seed_sequence, get_embeddings, find_closest

InteractiveShell.ast_node_interactivity = 'all'

def read_file(file_name):
    data = pd.read_csv(file_name)
    print(data.head())
    training_dict, word_idx, idx_word, sequences = get_data(file_name, training_len=50)

def main():
    read_file('./data/neural_network_patent_query.csv')

main()

