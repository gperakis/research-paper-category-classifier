import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from more_itertools import flatten

from rpcc.create_features import create_node2vec_embeddings_from_texts, TextFeaturesExtractor
from rpcc.load_data import DataLoader

EMB_DIM = 300


def prepare_node2vec_inputs(x: pd.Series, embedding_dim: int = EMB_DIM):
    """


    :param x:
    :param embedding_dim:
    :return:
    """
    x_expanded = x.apply(TextFeaturesExtractor.expand_contractions)
    x_word_tokens = x_expanded.apply(text_to_word_sequence)

    word_index = {word: idx + 1 for idx, word in enumerate(set(flatten(x_word_tokens)) | {'<UNK>'})}

    x_int_sequences = x_word_tokens.apply(lambda l: [word_index.get(word, word_index['<UNK>']) for word in l])

    max_seq_len = max([len(token_list) for token_list in x_int_sequences])

    train_data = pad_sequences(x_int_sequences, maxlen=max_seq_len)

    # constructing an embedding matrix
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(loc=0.0, scale=0.1, size=embedding_dim)

    return dict(word_index=word_index,
                maxlen=max_seq_len,
                data_seq=train_data,
                embedding_matrix=embedding_matrix)


if __name__ == "__main__":
    dl_obj = DataLoader()
    dl_obj.run_data_preparation()

    embeddings_index = create_node2vec_embeddings_from_texts(texts=[],
                                                             window_size=3,
                                                             emb_size=EMB_DIM,
                                                             filename='glove.abstracts.nodes',
                                                             load_embeddings=True,
                                                             save_embeddings=False)

    x_train = dl_obj.x_train['abstract']
    y_train = dl_obj.y_train

    tfe_obj = TextFeaturesExtractor(input_data=x_train)

    meta = prepare_node2vec_inputs(x_train)

    print(meta['word_index'].keys())
    print(meta['maxlen'])
    print(meta['data_seq'].head())
    print(meta['embedding_matrix'])
