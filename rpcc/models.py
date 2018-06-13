from keras import layers
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
import numpy as np
from pprint import pprint
from keras import regularizers

from keras import backend as K


class MyModel:
    def __init__(self,
                 emb_size: int,
                 voc_size: int,
                 max_sequence_length: int):
        """
        MyModel class is the parent class in model implementation. It implements the fit of the model to the data and
        stores the trained model and learning history. Also, it gives access to the data of a given layer.

        :param emb_size: int, the length of the word embeddings
        :param voc_size: int, the unique tokens of the corpus
        :param max_sequence_length: int, the max number of tokens of the input the sentence
        """
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.max_sequence_length = max_sequence_length

        self.model: Model = None
        self.history: list = None

    def build_model(self):
        """
        Abstract method implements model building with Keras
        """
        return None

    def fit(self, X, y, epochs: int = 10):
        """
        It fits a Keras model tot eh given data and returns the learning history.

        :param X: Numpy array of training data (if the model has a single input),
                or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output),
                or list of Numpy arrays (if the model has multiple outputs).
        :param epochs: Integer. Number of epochs to train the model.
        :return: A `History` object. Its `History.history` attribute is
                a record of training loss values and metrics values
                at successive epochs, as well as validation loss values
                and validation metrics values.
        """
        self.history = self.model.fit(X,
                                      y,
                                      epochs=epochs,
                                      batch_size=128)
        return self.history

    def predict(self):
        """
        Abstract method implements model prediction with Keras
        """
        pass

    def get_layer_values(self, X, layer_index=-2):
        """
        It accesses the given layer of object's model and returns the output of this layer.

        :param X: Numpy array of training data (if the model has a single input),
                or list of Numpy arrays (if the model has multiple inputs).
        :param layer_index: Integer. The index of the interested layer.
        :return: Numpy array with the output values of network's interested layer.
        """

        inp = self.model.input  # input placeholder
        outputs = [self.model.layers[layer_index].output]  # all layer outputs

        functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
        layer_outs = functor([X, 1.])

        return layer_outs[0]


class AbstractEmbedding(MyModel):
    def __init__(self,
                 emb_size: int,
                 voc_size: int,
                 max_sequence_length: int):
        """
        AbstractEmbedding class implements an LSTM model on a given corpus.

        We feed to the model of this class data of the abstracts of research papers. For each observation,
        we have transformed the corpus to an array of integers (output of Keras tokenizer).
        The labels of the observations stand for the category of the domain of the paper.

        :param emb_size: int, the length of the word embeddings
        :param voc_size: int, the unique tokens of the corpus
        :param max_sequence_length: int, the max number of tokens of the input the sentence
        """
        super().__init__(emb_size, voc_size, max_sequence_length)

    def build_model(self):
        """
        Creates and compiles an lstm model with Keras deep learning library
        """

        # define model
        text_input = Input(shape=(self.max_sequence_length,), dtype='int32', name='text')
        embedded_text = layers.Embedding(self.voc_size, self.emb_size)(text_input)
        encoded_text1 = layers.LSTM(32, return_sequences=True)(embedded_text)
        encoded_text2 = layers.LSTM(32)(encoded_text1)

        category = layers.Dense(2, activation='sigmoid')(encoded_text2)

        model = Model(text_input, [category])

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])

        print(model.summary())

        self.model = model


class TitleEmbedding(MyModel):
    def __init__(self,
                 emb_size: int,
                 voc_size: int,
                 max_sequence_length: int):
        """
        TitleEmbedding class implements an LSTM model on a given corpus.

        We feed to the model of this class data of the titles of research papers. For each observation,
        we have transformed the title corpus to an array of integers (output of Keras tokenizer).
        The labels of the observations stand for the category of the domain of the paper.

        :param emb_size: int, the length of the word embeddings
        :param voc_size: int, the unique tokens of the corpus
        :param max_sequence_length: int, the max number of tokens of the input the sentence
        """
        super().__init__(emb_size, voc_size, max_sequence_length)

    def build_model(self):
        """
        Creates and compiles an lstm model with Keras deep learning library
        """

        # define model
        text_input = Input(shape=(self.max_sequence_length,), dtype='int32', name='text')
        embedded_text = layers.Embedding(self.voc_size, self.emb_size)(text_input)
        encoded_text1 = layers.LSTM(16, return_sequences=True)(embedded_text)
        encoded_text2 = layers.LSTM(16)(encoded_text1)

        category = layers.Dense(2, activation='sigmoid')(encoded_text2)

        model = Model(text_input, [category])

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])

        print(model.summary())

        self.model = model


class FeedForward(MyModel):
    def __init__(self,
                 emb_size: int,
                 voc_size: int,
                 max_sequence_length: int):
        """
        FeedForward class implements a feed forward model with the use of Dense layers of Keras deep learning library.

        :param emb_size: int, the length of the word embeddings
        :param voc_size: int, the unique tokens of the corpus
        :param max_sequence_length: int, the max number of tokens of the input the sentence
        """
        super().__init__(emb_size, voc_size, max_sequence_length)

    def build_model(self):
        """
        Creates and compiles an lstm model with Keras deep learning library
        """

        # define model
        graph_input = Input(shape=(self.max_sequence_length,), dtype='float32', name='graph')
        abstract_input = Input(shape=(self.max_sequence_length,), dtype='float32', name='abstract')
        title_input = Input(shape=(self.max_sequence_length,), dtype='float32', name='title')
        author_input = Input(shape=(self.max_sequence_length,), dtype='float32', name='author')

        merged_input = layers.concatenate([graph_input, abstract_input, title_input, author_input], axis=-1)

        deep1 = layers.Dense(128, activation='relu')(merged_input)
        deep2 = layers.Dense(64, activation='relu')(deep1)
        category = layers.Dense(28, activation='softmax')(deep2)

        model = Model([graph_input, abstract_input, title_input, author_input], category)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['acc'])

        print(model.summary())

        self.model = model


class KCoreEmbedding(MyModel):
    def __init__(self, emb_size: int, voc_size: int, max_sequence_length: int):
        super().__init__(emb_size, voc_size, max_sequence_length)

    def build_model(self):
        pass


def build_model(inputs: list,
                dropout: float = 0.2,
                regularizer: tuple = ('l2', 0.01)):
    """

    :param inputs:
    :param dropout:
    :param regularizer:
    :return:
    """
    model_inputs_list = list()

    if regularizer[0] == 'l2':
        kernel_reg = regularizers.l2(regularizer[1])
    else:
        kernel_reg = regularizers.l1(regularizer[1])

    for doc in inputs:
        model_inputs_list.append(Input(shape=(doc['length'],),
                                       dtype='float32',
                                       name=doc['input_name']))

    merged_input = layers.concatenate(model_inputs_list, axis=-1)

    deep1 = layers.Dense(128,
                         activation='relu',
                         kernel_initializer='glorot_normal',
                         kernel_regularizer=kernel_reg)(merged_input)
    deep1 = layers.BatchNormalization()(deep1)
    deep1 = layers.Dropout(dropout)(deep1)

    deep2 = layers.Dense(64, activation='relu',
                         kernel_initializer='glorot_normal',
                         kernel_regularizer=kernel_reg)(deep1)

    deep2 = layers.BatchNormalization()(deep2)
    deep2 = layers.Dropout(dropout)(deep2)

    category = layers.Dense(28, activation='softmax')(deep2)

    model = Model(model_inputs_list, category)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    print(model.summary())

    return model


if __name__ == '__main__':
    text_vocabulary_size = 10000
    num_samples = 100
    max_length = 100

    # X_ = np.random.randint(1,
    #                        text_vocabulary_size,
    #                        size=(num_samples, max_length))
    #
    # y_ = to_categorical(np.random.randint(low=0, high=2, size=num_samples))
    #
    # obj = AbstractEmbedding(emb_size=10,
    #                         voc_size=text_vocabulary_size,
    #                         max_sequence_length=max_length)
    #
    # obj.build_model()
    #
    # obj.fit(X_, y_, 15)
    #
    # asdf = obj.get_layer_values(X=X_)
    #
    # pprint(asdf)
    #
    # pprint(asdf.shape)

    obj = FeedForward(10, 10000, 100)

    obj.build_model()

    model_inputs = [
        {'input_name': 'graph', 'length': 50, 'weight': 1},
        {'input_name': 'abstract', 'length': 50, 'weight': 1},
        {'input_name': 'title', 'length': 50, 'weight': 1},
        {'input_name': 'author', 'length': 50, 'weight': 1}]

    x = build_model(inputs=model_inputs)
