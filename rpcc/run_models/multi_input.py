import numpy as np
from keras import layers
from keras import regularizers
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer

from rpcc.create_features import TextFeaturesExtractor, GraphFeaturesExtractor
from rpcc.load_data import DataLoader


# from keras.models import load_model


def get_citations_embeddings(citations_ids: list,
                             embeddings_dict: dict,
                             emb_size: int = 200) -> np.array:
    """

    :param citations_ids:
    :param embeddings_dict:
    :param emb_size:
    :return:
    """
    print('getting text_centroid_embeddings')
    citation_vectors = list()

    for citation in citations_ids:
        token_emb = embeddings_dict.get(citation, None)
        if token_emb is None:
            citation_vectors.append(np.random.normal(loc=0.0,
                                                     scale=0.1,
                                                     size=emb_size))

        else:
            citation_vectors.append(token_emb)

    return np.array(citation_vectors)


# loading data
dl_obj = DataLoader()
dl_obj.run_data_preparation()

# creating a label binarizer instance in order to convert the classes to one hot vectors
lb = LabelBinarizer()

# extracting the train_val targets
y_train_val = dl_obj.y_train_validation
# converting the train_val targets to one hot
y_train_val_one_hot = lb.fit_transform(y_train_val)

# checking the validity of the shape
print(y_train_val_one_hot.shape)

# extracting the train_val abstracts
x_train_val_abstract = dl_obj.x_train_validation['abstract']
# extracting the train_val titles
x_train_val_title = dl_obj.x_train_validation['title']

x_train_val_ids = dl_obj.train_validation_ids
# extracting the test abstracts
x_test_abstract = dl_obj.x_test['abstract']
# extracting the test titles
x_test_title = dl_obj.x_test['title']

# instantiating the text features class
tfe_obj = TextFeaturesExtractor(input_data=None)

# creating separate tokenizers for titles and abstracts, creating metadata for each of the two.
abstracts_meta = tfe_obj.pre_process_text(texts=x_train_val_abstract,
                                          remove_stopwords=True)
titles_meta = tfe_obj.pre_process_text(texts=x_train_val_title,
                                       remove_stopwords=True)

# abstracts padded
x_train_val_abstracts_padded = abstracts_meta['x']
# the following meta are needed for the model that we will train
x_train_val_abstracts_int2word = abstracts_meta['int2word']
x_train_val_abstracts_word2int = abstracts_meta['word2int']
x_train_val_abstracts_max_length = abstracts_meta['max_length']
x_train_val_abstracts_tokenizer = abstracts_meta['tokenizer']

# we also use the same tokenizer and padding for the test abstracts too.
x_test_abstracts_padded = tfe_obj.text_to_padded_sequences(texts=x_test_abstract,
                                                           tokenizer=x_train_val_abstracts_tokenizer,
                                                           max_length=x_train_val_abstracts_max_length)
# titles padded
x_train_val_titles_padded = titles_meta['x']
# the following meta are needed for the model that we will train
x_train_val_titles_int2word = titles_meta['int2word']
x_train_val_titles_word2int = titles_meta['word2int']
x_train_val_titles_max_length = titles_meta['max_length']
x_train_val_titles_tokenizer = titles_meta['tokenizer']

# we also use the same tokenizer and padding for the test titles too.
x_test_title_padded = tfe_obj.text_to_padded_sequences(texts=x_test_abstract,
                                                       tokenizer=x_train_val_titles_tokenizer,
                                                       max_length=x_train_val_titles_max_length)

gfe_obj = GraphFeaturesExtractor(graph=dl_obj.citation_graph)

NODE_2_VEC_EMB_SIZE = 300
citation_graph_emb_dict = gfe_obj.create_node2vec_embeddings(
    emb_size=NODE_2_VEC_EMB_SIZE,
    filename='glove.citation.graph.nodes',
    load_embeddings=True)

x_train_val_citations_emb = get_citations_embeddings(citations_ids=dl_obj.train_validation_ids,
                                                     embeddings_dict=citation_graph_emb_dict,
                                                     emb_size=NODE_2_VEC_EMB_SIZE)

x_test_citations_emb = get_citations_embeddings(citations_ids=dl_obj.test_ids,
                                                embeddings_dict=citation_graph_emb_dict,
                                                emb_size=NODE_2_VEC_EMB_SIZE)

print(x_train_val_citations_emb.shape)
print(x_test_citations_emb.shape)

citations_node_metrics = gfe_obj.create_simple_node_features(load_metrics=True,
                                                             save_metrics=False,
                                                             outfile='citation_graph_simple_metrics.csv')

citations_node_metrics['node'] = citations_node_metrics.node.apply(str)
citations_node_metrics.set_index('node', inplace=True)

x_train_val_citation_metrics = citations_node_metrics.loc[dl_obj.train_validation_ids]
x_test_citation_metrics = citations_node_metrics.loc[dl_obj.test_ids]

############################################################################

x_train_val_static = np.concatenate((x_train_val_citation_metrics.values,
                                     x_train_val_citations_emb), axis=1)

x_test_static = np.concatenate((x_test_citation_metrics.values,
                                x_test_citations_emb), axis=1)

print(x_train_val_static.shape)
print(x_test_static.shape)

############################################################################
############################################################################
############################################################################

dropout = 0.5
RNN_EMB_SIZE = 300
RNN_SIZE = 50
STATIC_INPUT_SIZE = x_train_val_static.shape[1]
lr = 0.001
regularization = regularizers.l2(0.01)
N_EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_SIZE = 0.1

abstracts_voc_size = len(x_train_val_abstracts_int2word)
titles_voc_size = len(x_train_val_titles_int2word)

# define model

# abstract part
abstract_input = Input(shape=(x_train_val_abstracts_max_length,), dtype='int32', name='abstract_input')
embedded_abstract_text = layers.Embedding(abstracts_voc_size, RNN_EMB_SIZE)(abstract_input)
encoded_abstract_text = layers.LSTM(RNN_SIZE, return_sequences=False,
                                    dropout=dropout, recurrent_dropout=dropout)(embedded_abstract_text)

# title part
title_input = Input(shape=(x_train_val_titles_max_length,), dtype='int32', name='title_input')
embedded_title_text = layers.Embedding(titles_voc_size, RNN_EMB_SIZE)(title_input)
encoded_title_text = layers.LSTM(RNN_SIZE, return_sequences=False,
                                 dropout=dropout, recurrent_dropout=dropout)(embedded_title_text)

# metrics part
metrics_input = Input(shape=(STATIC_INPUT_SIZE,), dtype='float32', name='metrics_input')
metrics_deep_1 = layers.Dense(128,
                              activation='tanh',
                              kernel_initializer='glorot_normal',
                              kernel_regularizer=regularization)(metrics_input)
metrics_deep_1 = layers.BatchNormalization()(metrics_deep_1)
metrics_deep_1 = layers.Dropout(dropout)(metrics_deep_1)

merged_layer = layers.concatenate([encoded_abstract_text,
                                   encoded_title_text,
                                   metrics_deep_1], axis=-1)

merged_layer_1 = layers.Dense(64,
                              activation='tanh',
                              kernel_initializer='glorot_normal',
                              kernel_regularizer=regularization)(merged_layer)

merged_layer_1 = layers.BatchNormalization()(merged_layer_1)
merged_layer_1 = layers.Dropout(dropout)(merged_layer_1)

category = layers.Dense(28, activation='softmax', name='main_output')(merged_layer_1)

model = Model([abstract_input, title_input, metrics_input], category)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

print(model.summary())

opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

history = model.fit(x=[x_train_val_abstracts_padded,
                       x_train_val_titles_padded,
                       x_train_val_static],
                    y=y_train_val_one_hot,
                    epochs=N_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=VALIDATION_SIZE,
                    verbose=2)
