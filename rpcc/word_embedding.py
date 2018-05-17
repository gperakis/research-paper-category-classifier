import os

import fasttext
import numpy as np
import pandas as pd
from tqdm import tqdm

from rpcc import DATA_DIR, MODELS_DIR


class GloveWordEmbedding:
    def __init__(self):
        pass

    @staticmethod
    def get_word_embeddings(dimension=200):
        """
        This method reads the
        :return: dict. with the vocabulary word and its word embedding vector in a list
        """
        assert dimension in [50, 100, 200, 300]

        t = 'glove.6B.{}d.txt'.format(dimension)

        print('Loading Word Embeddings file: {}'.format(t))
        infile = os.path.join(DATA_DIR, t)

        with open(infile, 'rb') as in_file:
            text = in_file.read().decode("utf-8")

        word_embeddings = dict()
        for line in tqdm(text.split('\n'),
                         desc='Loading Embeddings for {} dimensions'.format(dimension),
                         unit=' Embeddings'):
            try:
                w_e_numbers = list(map(lambda x: float(x), line.split()[1:]))
                word_embeddings[line.split()[0]] = w_e_numbers
            except IndexError:
                pass

        return word_embeddings

    @staticmethod
    def get_word_embeddings_mean(dimension=50, save_data=False, load_data=True):
        """
        This method reads the embeddings, calculates the mean for every embedding and creates a mapper
        :param dimension:
        :param save_data:
        :param load_data:
        :return: dict. with the vocabulary word and its word embedding vector in a list
        """
        assert dimension in [50, 100, 200, 300]

        if load_data:
            try:
                t = 'glove.6B.{}d_mean.csv'.format(dimension)
                infile = "{}{}".format(DATA_DIR, t)
                print('Loading file: {}'.format(infile))
                df = pd.read_csv(infile)

                return df.set_index(['word']).to_dict().get('mean_embedding')

            except FileNotFoundError:
                print('File Not Found in specified Directory. Creating a new one from scratch')

        t = 'glove.6B.{}d.txt'.format(dimension)

        print('Loading Word Embeddings file: {}'.format(t))
        infile = os.path.join(DATA_DIR, t)

        with open(infile, 'rb') as in_file:
            text = in_file.read().decode("utf-8")

        word_embeddings = list()
        for line in tqdm(text.split('\n'),
                         desc='Loading Embeddings for {} dimensions'.format(dimension),
                         unit=' Embeddings'):
            try:
                w_e_numbers = list(map(lambda x: float(x), line.split()[1:]))

                word_embeddings.append((line.split()[0], np.mean(w_e_numbers)))
            except IndexError:
                pass

        mean_embeddings_df = pd.DataFrame(word_embeddings, columns=['word', 'mean_embedding'])

        if save_data:
            t = 'glove.6B.{}d_mean.csv'.format(dimension)
            outfile = "{}{}".format(DATA_DIR, t)
            mean_embeddings_df.to_csv(outfile, encoding='utf-8', index=False)

        return mean_embeddings_df.set_index(['word']).to_dict().get('mean_embedding')


class FastTextEmbedding:
    def __init__(self,
                 embedding_type='skipgram',
                 model_name='model'):
        """
        :param embedding_type:
        :param model_name:
        """
        assert embedding_type in ['skipgram', 'cbow']

        self.embedding_type = embedding_type
        self.model_name = model_name
        self.model_path = os.path.join(MODELS_DIR, '{}_{}'.format(self.embedding_type, model_name))
        self.model = None

    def train_model(self, file_input='train_text.txt'):
        """
        :param file_input:
        :return:
        """
        input_path = os.path.join(DATA_DIR, file_input)

        if self.embedding_type == 'skipgram':
            # Skipgram model
            model = fasttext.skipgram(input_path, self.model_path)

        elif self.embedding_type == 'cbow':
            # CBOW model
            model = fasttext.cbow(input_path, self.model_path)

        else:
            raise NotImplementedError()

        print(model.words)  # list of words in dictionary

        self.model = model

        return model

    def load_model(self):
        """
        :return:
        """
        filepath = '{}.bin'.format(self.model_path)
        self.model = fasttext.load_model(filepath)
        return self.model

    def load_pretrained(self, path='wiki-news-300d-1M.vec'):
        """
        :param path:
        :return:
        """

        filepath = os.path.join(MODELS_DIR, path)
        self.model = fasttext.load_model(filepath)

        return self.model


if __name__ == '__main__':
    fte_obj = FastTextEmbedding(embedding_type='skipgram')

    trained_model = fte_obj.load_pretrained()
    print(len(trained_model.trained_model))
