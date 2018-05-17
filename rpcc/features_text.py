import re

import numpy as np
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm

from rpcc import setup_logger, CONTRACTION_MAP
from rpcc.text_mining import tokenize_text
from rpcc.word_embedding import GloveWordEmbedding

SPACY_NLP = spacy.load('en', parse=False, tag=False, entity=False)

logger = setup_logger(__name__)


class ModelTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts a number of columns and return these columns"""

    def __init__(self, columns):
        """

        :param columns:
        """
        self.columns = columns

    def transform(self, X, y=None):

        if set(self.columns).issubset(set(X.columns.tolist())):
            return X[self.columns].values

        else:
            raise Exception('Columns declared, not in dataframe')

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""

        return self


class TextColumnExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts the column with the text"""

    def __init__(self, column):
        """

        :param column:
        """
        self.column = column

    def transform(self, X, y=None):

        if {self.column}.issubset(set(X.columns.tolist())):
            return X[self.column]

        else:
            raise Exception('Columns declared, not in dataframe')

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""

        return self


class DenseTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


class SingleColumnDimensionReshaper(BaseEstimator, TransformerMixin):

    def __init__(self):
        """

        """
        pass

    def transform(self, X, y=None):
        return X.values.reshape(-1, 1)

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""

        return self


class WordLengthMetricsExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts text column, splits text in tokens and outputs average word length"""

    def __init__(self,
                 col_name,
                 split_type='simple',
                 metric='avg',
                 reshape=True):
        """

        :param col_name:
        :param split_type:
        :param metric:
        :param reshape:
        """
        assert metric in ['avg', 'std']
        self.split_type = split_type
        self.col_name = col_name
        self.metric = metric
        self.reshape = reshape

    def calculate_metric(self, words):
        """
        Helper code to compute average word length of a name
        :param words:
        :return:
        """
        if words:
            if self.metric == 'avg':
                return np.mean([len(word) for word in words])

            elif self.metric == 'std':
                return np.std([len(word) for word in words])

        else:
            return 0

    def transform(self, X, y=None):

        if X is None:
            x = X.apply(lambda s: tokenize_text(text=s, split_type=self.split_type))

        else:
            logger.info('Calculating {} for "{}" Column'.format(self.metric, self.col_name))
            x = X[self.col_name].apply(lambda s: tokenize_text(text=s, split_type=self.split_type))

        out = x.apply(self.calculate_metric)

        if self.reshape:
            return out.values.reshape(-1, 1)

        return out

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts text column, returns sentence's length"""

    def __init__(self, col_name, reshape=True):
        """

        :param col_name:
        :param reshape:
        """
        self.col_name = col_name
        self.reshape = reshape

    def transform(self, X, y=None):
        if X is None:
            logger.info('Calculating text length for "{}" Column'.format(self.col_name))
            out = X.apply(len)

        else:
            logger.info('Calculating text length for "{}" Column'.format(self.col_name))
            out = X[self.col_name].apply(len)

        if self.reshape:
            return out.values.reshape(-1, 1)

        return out

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class NumberOfTokensCalculator(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts number of tokens in text"""

    def __init__(self, col_name, reshape=True):
        """
        :param col_name:
        """
        self.col_name = col_name
        self.reshape = reshape

    def transform(self, X, y=None):
        logger.info('Counting number of tokens for "{}" Column'.format(self.col_name))

        if X is None:
            out = X.apply(lambda x: len(tokenize_text(x, split_type='thorough')))

        else:
            out = X[self.col_name].apply(lambda x: len(tokenize_text(x, split_type='thorough')))

        if self.reshape:
            return out.values.reshape(-1, 1)

        return out

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class SentenceEmbeddingExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, the average of sentence's word embeddings"""

    def __init__(self,
                 col_name=None,
                 embedding_type='tf',
                 embedding_dimensions=50,
                 word_embeddings_dict=None):
        """

        :param col_name:
        :param embedding_type:
        :param embedding_dimensions:
        """
        assert embedding_type in ['tf', 'tfidf']
        assert embedding_dimensions in [50, 100, 200, 300]

        self.col_name = col_name
        self.embedding_dimensions = embedding_dimensions
        self.embedding_type = embedding_type
        self.word_embeddings_dict = word_embeddings_dict

    def calculate_updated_sentence_embeddings(self, X):
        """

        :param X:
        :return:
        """

        if self.word_embeddings_dict is None:
            logger.info('Loading word embeddings for {} dimensions'.format(self.embedding_dimensions))
            word_embeddings = GloveWordEmbedding.get_word_embeddings(dimension=self.embedding_dimensions)
        else:
            logger.info('Loading pre loaded word embeddings for {} dimensions'.format(self.embedding_dimensions))
            word_embeddings = self.word_embeddings_dict.get(self.embedding_dimensions)

        if self.embedding_type == 'tf':

            vectorizer = CountVectorizer(strip_accents='unicode',
                                         analyzer='word',
                                         ngram_range=(1, 1),
                                         stop_words=None,
                                         lowercase=True,
                                         binary=False)

        elif self.embedding_type == 'tfidf':

            vectorizer = TfidfVectorizer(strip_accents='unicode',
                                         analyzer='word',
                                         ngram_range=(1, 1),
                                         stop_words=None,
                                         lowercase=True,
                                         binary=False,
                                         norm='l2',
                                         use_idf=True,
                                         smooth_idf=True)

        else:
            raise NotImplementedError()

        X_transformed = vectorizer.fit_transform(X)

        analyser = vectorizer.build_analyzer()
        vocabulary_indices = vectorizer.vocabulary_

        embedded_vectors_updated = list()

        for index_row, doc in enumerate(tqdm(X, unit=' Document')):

            doc_vector = np.zeros(self.embedding_dimensions, dtype=float)
            sum_of_tf_or_idfs = 0

            # breaks test in tokens.
            doc_tokens = analyser(doc)

            # We keep only the unique ones in order to get the tf-idf values from the stored matrix X_transformed.
            for token in set(doc_tokens):
                # get column index from the vocabulary in order to find the exact spot in the X_transformed matrix
                index_col = vocabulary_indices[token]

                # Getting the tf or idf value for the given word from the transformed matrix
                token_tf_or_idf_value = X_transformed[index_row, index_col]

                # search for the embedding vector of the given token. If not found processed a vector of zeros
                # with the same shape.
                token_embedding_vector = word_embeddings.get(token,
                                                             np.zeros(self.embedding_dimensions, dtype=float))

                # Calculating the element product of the idf and embedding vector
                token_embedding_vector = np.multiply(token_embedding_vector, token_tf_or_idf_value)

                sum_of_tf_or_idfs += token_tf_or_idf_value

                doc_vector += token_embedding_vector

            doc_final_vector = np.divide(doc_vector, sum_of_tf_or_idfs)

            embedded_vectors_updated.append(doc_final_vector)

        return np.vstack(embedded_vectors_updated)

    def transform(self, X, y=None):

        if self.col_name is None:
            logger.info('Calculating word embeddings of sentences for pandas series')
            return self.calculate_updated_sentence_embeddings(X=X)

        logger.info('Calculating word embeddings of sentences for "{}" Column'.format(self.col_name))
        return self.calculate_updated_sentence_embeddings(X=X[self.col_name])

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class ContractionsExpander(BaseEstimator, TransformerMixin):
    """Takes in data-frame, the average of sentence's word embeddings"""

    def __init__(self,
                 col_name=None,
                 contractions_mapper=CONTRACTION_MAP):
        """

        :param col_name:
        :param contractions_mapper:
        """
        self.col_name = col_name
        self.contractions_mapper = contractions_mapper

    def expand_contractions(self, text):
        """
        This function expands contractions for the english language. For example "I've" will become "I have".

        :param text:
        :return:
        """

        contractions_pattern = re.compile('({})'.format('|'.join(self.contractions_mapper.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            """
            This sub function helps into expanding a given contraction
            :param contraction:
            :return:
            """
            match = contraction.group(0)
            first_char = match[0]

            expanded_contr = self.contractions_mapper.get(
                match) if self.contractions_mapper.get(match) else self.contractions_mapper.get(match.lower())

            expanded_contr = first_char + expanded_contr[1:]

            return expanded_contr

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)

        return expanded_text

    def transform(self, X, y=None):
        if self.col_name is None:
            logger.info('Extracting contractions for pandas series')
            return X.apply(self.expand_contractions)

        logger.info('Extracting contractions for "{}" Column'.format(self.col_name))
        return X[self.col_name].apply(self.expand_contractions)

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class LemmaExtractor(BaseEstimator, TransformerMixin):
    """Takes in data-frame, gets lemmatized words"""

    def __init__(self,
                 col_name=None,
                 spacy_nlp=SPACY_NLP,
                 contractions_mapper=CONTRACTION_MAP):
        """

        :param col_name:
        :param spacy_nlp:
        :param contractions_mapper:
        """
        self.col_name = col_name
        self.contractions_mapper = contractions_mapper
        self.spacy_nlp = spacy_nlp

    def lemmatize_text(self, text):
        """
        This method lemmatizes text
        :param text:
        :return:
        """
        text = self.spacy_nlp(text)

        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

        return text

    def transform(self, X, y=None):
        if self.col_name is None:
            logger.info('Extracted Lemmatized words for text for pandas series')
            return X.apply(self.lemmatize_text)

        logger.info('Extracted Lemmatized words for text for "{}" Column'.format(self.col_name))
        return X[self.col_name].apply(self.lemmatize_text)

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
