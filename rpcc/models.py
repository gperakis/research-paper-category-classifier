import os

import numpy as np
import pandas as pd
from keras import backend as K
from keras import layers
from keras import regularizers
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from rpcc import DATA_DIR
from keras import regularizers


class ModelNN:
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

        self.model = None
        self.history = None
        self.features = None

    def build_model(self):
        """
        Abstract method implements model building with Keras
        """
        pass

    def fit(self, X, y, epochs: int = 10, val_size=0.2, bs=128, lr=0.001):
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
        opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.model.compile(optimizer=opt,
                           loss='categorical_crossentropy',
                           metrics=['acc'])

        self.history = self.model.fit(X,
                                      y,
                                      epochs=epochs,
                                      batch_size=bs,
                                      validation_split=val_size,
                                      verbose=2)
        return self.history

    def predict(self, X, y, dump=False):
        """
        Abstract method implements model prediction with Keras
        """
        scores = self.model.evaluate(x=X,
                                     y=y,
                                     verbose=1)

        predicted_classes = self.model.predict(X)

        pred_scores = np.squeeze(predicted_classes)

        predicted_classes = list(map(lambda x: 1 if x > 0.5 else 0, list(pred_scores)))

        print('predicted_classes: {}'.format(predicted_classes))

        if dump:
            # create a DataFrame with the predictions and write them to csv
            prediction_df = pd.DataFrame(data=X)
            prediction_df = prediction_df.assign(predictions=pd.Series(predicted_classes).values)

            outfile_path = os.path.join(DATA_DIR, 'predictions.csv')
            prediction_df.to_csv(outfile_path, sep='\t')

        return {'scores': {'loss': scores[0],
                           'acc': scores[1]},
                'y_pred': predicted_classes,
                'y_pred_scores': pred_scores,
                'y_true': y}

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


class AbstractEmbedding(ModelNN):
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

    def build_model(self, dropout=0.2, rnn_size=50):
        """
        Creates and compiles an lstm model with Keras deep learning library
        """

        # define model
        text_input = Input(shape=(self.max_sequence_length,), dtype='int32', name='text')
        embedded_text = layers.Embedding(self.voc_size, self.emb_size)(text_input)
        encoded_text1 = layers.LSTM(rnn_size,
                                    return_sequences=True,
                                    dropout=dropout,
                                    recurrent_dropout=dropout)(embedded_text)
        encoded_text2 = layers.LSTM(rnn_size,
                                    dropout=dropout,
                                    recurrent_dropout=dropout)(encoded_text1)

        category = layers.Dense(28, activation='softmax')(encoded_text2)

        model = Model(text_input, [category])

        print(model.summary())

        self.model = model


class CNN(ModelNN):
    """

    """

    def __init__(self, emb_size: int,
                 voc_size: int,
                 max_sequence_length: int):
        """

        :param emb_size:
        :param voc_size:
        :param max_sequence_length:
        """
        super().__init__(emb_size, voc_size, max_sequence_length)

    def build_model(self, dropout=0.2, deep_activ='relu'):
        """
        Creates and compiles a cnn model with Keras deep learning library

        """
        assert deep_activ in ['relu', 'tanh']

        text_input = Input(shape=(self.max_sequence_length,), dtype='int32', name='text')
        embedded_text = layers.Embedding(self.voc_size,
                                         self.emb_size,
                                         embeddings_regularizer=regularizers.l2(0.001))(text_input)

        l_cov1 = layers.Conv1D(64, 2, activation=deep_activ)(embedded_text)
        l_drop1 = layers.Dropout(dropout)(l_cov1)
        l_pool1 = layers.MaxPooling1D(2)(l_drop1)

        l_cov2 = layers.Conv1D(64, 2,
                               activation=deep_activ)(l_pool1)
        l_drop2 = layers.Dropout(dropout)(l_cov2)
        l_pool2 = layers.MaxPooling1D(2)(l_drop2)

        l_cov3 = layers.Conv1D(64, 2,
                               activation=deep_activ)(l_pool2)
        l_drop3 = layers.Dropout(dropout)(l_cov3)
        l_pool3 = layers.MaxPooling1D(2)(l_drop3)  # global max pooling

        l_flat = layers.Flatten()(l_pool3)
        l_dense = layers.Dense(64, activation='relu')(l_flat)
        l_drop_dense = layers.Dropout(dropout)(l_dense)

        category = layers.Dense(28, activation='softmax')(l_drop_dense)

        model = Model(text_input, [category])

        self.model = model

        print(model.summary())


class FeedForward(ModelNN):
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
        self.build_model()

    def build_model(self):
        """
        Creates and compiles an lstm model with Keras deep learning library
        """

        # define model
        # graph_input = Input(shape=(self.max_sequence_length,), dtype='float32', name='graph')
        # abstract_input = Input(shape=(self.max_sequence_length,), dtype='float32', name='abstract')
        # title_input = Input(shape=(self.max_sequence_length,), dtype='float32', name='title')
        # author_input = Input(shape=(self.max_sequence_length,), dtype='float32', name='author')

        # merged_input = layers.concatenate([graph_input, abstract_input, title_input, author_input], axis=-1)

        graph_input = Input(shape=(self.max_sequence_length,), dtype='float32', name='graph')

        deep1 = layers.Dense(128, activation='relu')(graph_input)
        deep3 = layers.Dense(64, activation='relu')(deep1)
        category = layers.Dense(28, activation='softmax')(deep3)

        # model = Model([graph_input, abstract_input, title_input, author_input], category)
        model = Model([graph_input], category)

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['acc'])

        print(model.summary())

        self.model = model


class KCoreEmbedding(ModelNN):
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
