import os

import numpy as np
from keras import callbacks
from keras import layers
from keras import regularizers
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
import os
from rpcc import PROCESSED_DATA_DIR
import pandas as pd
from rpcc import MODELS_DIR
from rpcc import TENSORBOARD_LOGS_DIR
from rpcc.create_features import TextFeaturesExtractor, GraphFeaturesExtractor, get_authors_community_vectors
from rpcc.load_data import DataLoader
import networkx as nx
from grakel import GraphKernel

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
dl_obj = DataLoader(verbose=0)
dl_obj.run_data_preparation(val_size=0.1, random_state=5)

# creating a label binarizer instance in order to convert the classes to one hot vectors
lb = LabelBinarizer()

# extracting the train targets
y_train = dl_obj.y_train
# converting the train targets to one hot
y_train_one_hot = lb.fit_transform(y_train)

# extracting the train targets
y_val = dl_obj.y_val
# converting the train targets to one hot
y_val_one_hot = lb.transform(y_val)

# checking the validity of the shape
print('Y train one hot shape: ', y_train_one_hot.shape)
print('Y val one hot shape: ', y_val_one_hot.shape)

# extracting the train_val abstracts
x_train_abstract = dl_obj.x_train['abstract']
# extracting the train_val titles
x_train_title = dl_obj.x_train['title']

# extracting the train_val abstracts
x_val_abstract = dl_obj.x_val['abstract']
# extracting the train_val titles
x_val_title = dl_obj.x_val['title']

# extracting the test abstracts
x_test_abstract = dl_obj.x_test['abstract']
# extracting the test titles
x_test_title = dl_obj.x_test['title']

# instantiating the text features class
tfe_obj = TextFeaturesExtractor(input_data=None)

# creating separate tokenizers for titles and abstracts, creating metadata for each of the two.
abstracts_meta = tfe_obj.pre_process_text(texts=x_train_abstract,
                                          remove_stopwords=True)

# abstracts padded
x_train_abstracts_padded = abstracts_meta['x']
# the following meta are needed for the model that we will train
x_train_abstracts_int2word = abstracts_meta['int2word']
x_train_abstracts_word2int = abstracts_meta['word2int']
x_train_abstracts_max_length = abstracts_meta['max_length']
x_train_abstracts_tokenizer = abstracts_meta['tokenizer']

# we also use the same tokenizer and padding for the validation abstracts too.
x_val_abstracts_padded = tfe_obj.text_to_padded_sequences(texts=x_val_abstract,
                                                          tokenizer=x_train_abstracts_tokenizer,
                                                          max_length=x_train_abstracts_max_length)

# we also use the same tokenizer and padding for the test abstracts too.
x_test_abstracts_padded = tfe_obj.text_to_padded_sequences(texts=x_test_abstract,
                                                           tokenizer=x_train_abstracts_tokenizer,
                                                           max_length=x_train_abstracts_max_length)

#####################################################################################################
#####################################################################################################
#####################################################################################################

titles_meta = tfe_obj.pre_process_text(texts=x_train_title,
                                       remove_stopwords=True)
# titles padded
x_train_titles_padded = titles_meta['x']
# the following meta are needed for the model that we will train
x_train_titles_int2word = titles_meta['int2word']
x_train_titles_word2int = titles_meta['word2int']
x_train_titles_max_length = titles_meta['max_length']
x_train_titles_tokenizer = titles_meta['tokenizer']

# we also use the same tokenizer and padding for the val titles too.
x_val_titles_padded = tfe_obj.text_to_padded_sequences(texts=x_val_title,
                                                       tokenizer=x_train_titles_tokenizer,
                                                       max_length=x_train_titles_max_length)

# we also use the same tokenizer and padding for the test titles too.
x_test_titles_padded = tfe_obj.text_to_padded_sequences(texts=x_test_title,
                                                        tokenizer=x_train_titles_tokenizer,
                                                        max_length=x_train_titles_max_length)

#####################################################################################################
#####################################################################################################
#####################################################################################################

gfe_obj = GraphFeaturesExtractor(graph=dl_obj.citation_graph)

NODE_2_VEC_EMB_SIZE = 300

citation_graph_emb_dict = gfe_obj.create_node2vec_embeddings(
    emb_size=NODE_2_VEC_EMB_SIZE,
    filename='glove.citation.graph.nodes',
    load_embeddings=True)

x_train_citations_emb = get_citations_embeddings(citations_ids=dl_obj.train_ids,
                                                 embeddings_dict=citation_graph_emb_dict,
                                                 emb_size=NODE_2_VEC_EMB_SIZE)

x_val_citations_emb = get_citations_embeddings(citations_ids=dl_obj.validation_ids,
                                               embeddings_dict=citation_graph_emb_dict,
                                               emb_size=NODE_2_VEC_EMB_SIZE)

x_test_citations_emb = get_citations_embeddings(citations_ids=dl_obj.test_ids,
                                                embeddings_dict=citation_graph_emb_dict,
                                                emb_size=NODE_2_VEC_EMB_SIZE)

print('X train citation embeddings shape: ', x_train_citations_emb.shape)
print('X val citation embeddings shape: ', x_val_citations_emb.shape)
print('X test citation embeddings shape: ', x_test_citations_emb.shape)

citations_node_metrics = gfe_obj.create_simple_node_features(load_metrics=True,
                                                             save_metrics=False,
                                                             outfile='citation_graph_simple_metrics.csv')

citations_node_metrics['node'] = citations_node_metrics.node.apply(str)
citations_node_metrics.set_index('node', inplace=True)

x_train_citation_metrics = citations_node_metrics.loc[dl_obj.train_ids]
x_val_citation_metrics = citations_node_metrics.loc[dl_obj.validation_ids]
x_test_citation_metrics = citations_node_metrics.loc[dl_obj.test_ids]

#####################################################################################################
#####################################################################################################
#####################################################################################################

community_outfile = os.path.join(PROCESSED_DATA_DIR, 'citation_communities.csv')

community_df = pd.read_csv(community_outfile)
community_df.rename(columns={'Unnamed: 0': 'article'}, inplace=True)
community_df['article'] = community_df['article'].apply(str)
community_df.set_index('article', inplace=True)

x_train_comm = community_df.loc[dl_obj.train_ids].values
x_val_comm = community_df.loc[dl_obj.validation_ids].values
x_test_comm = community_df.loc[dl_obj.test_ids].values

#####################################################################################################
#####################################################################################################
#####################################################################################################
# Authors communities metadata

authors_community_infile = os.path.join(PROCESSED_DATA_DIR, 'authors_communities.csv')

authors_community_df = pd.read_csv(authors_community_infile)
authors_community_df.rename(columns={'Unnamed: 0': 'authors'}, inplace=True)
authors_community_df['authors'].apply(lambda x: x.lower().strip())
authors_community_df.set_index('authors', inplace=True)

x_train_authors_communities = get_authors_community_vectors(authors_list=dl_obj.x_train['authors'],
                                                            authors_community_df=authors_community_df)

x_val_authors_communities = get_authors_community_vectors(authors_list=dl_obj.x_val['authors'],
                                                          authors_community_df=authors_community_df)

x_test_authors_communities = get_authors_community_vectors(authors_list=dl_obj.x_test['authors'],
                                                           authors_community_df=authors_community_df)

print('Authors community x_train size: ', x_train_authors_communities.shape)
print('Authors community x_val size: ', x_val_authors_communities.shape)
print('Authors community x_test size: ', x_test_authors_communities.shape)

######################################################################################################
######################################################################################################
######################################################################################################
# GRAPH KERNEL IMPLEMENTATION

REMOVE_STOP_WORDS = True
DIRECTED = False
sp_kernel = GraphKernel(kernel={"name": "shortest_path",
                                'with_labels': False},
                        normalize=True)

x_train_abstract_graphs = list()
for text in dl_obj.x_train['abstract']:
    graph = tfe_obj.generate_graph_from_text(text=text,
                                             remove_stopwords=REMOVE_STOP_WORDS,
                                             directed=DIRECTED)

    inp = nx.to_dict_of_lists(graph)
    x_train_abstract_graphs.append([inp])

x_val_abstract_graphs = list()
for text in dl_obj.x_val['abstract']:
    graph = tfe_obj.generate_graph_from_text(text=text,
                                             remove_stopwords=REMOVE_STOP_WORDS,
                                             directed=DIRECTED)

    inp = nx.to_dict_of_lists(graph)
    x_val_abstract_graphs.append([inp])

x_test_abstract_graphs = list()
for text in dl_obj.x_test['abstract']:
    graph = tfe_obj.generate_graph_from_text(text=text,
                                             remove_stopwords=REMOVE_STOP_WORDS,
                                             directed=DIRECTED)

    inp = nx.to_dict_of_lists(graph)
    x_test_abstract_graphs.append([inp])

K_train = sp_kernel.fit_transform(x_train_abstract_graphs)
K_val = sp_kernel.transform(x_val_abstract_graphs)
K_test = sp_kernel.transform(x_test_abstract_graphs)

######################################################################################################
#####################################################################################################
#####################################################################################################
x_train_static = np.concatenate((x_train_citation_metrics.values,
                                 x_train_citations_emb,
                                 x_train_comm,
                                 x_train_authors_communities,
                                 K_train), axis=1)

x_val_static = np.concatenate((x_val_citation_metrics.values,
                               x_val_citations_emb,
                               x_val_comm,
                               x_val_authors_communities,
                               K_val), axis=1)

x_test_static = np.concatenate((x_test_citation_metrics.values,
                                x_test_citations_emb,
                                x_test_comm,
                                x_test_authors_communities,
                                K_test), axis=1)

print('X train static metrics shape: ', x_train_static.shape)
print('X val static metrics shape: ', x_val_static.shape)
print('X test static metrics shape: ', x_test_static.shape)

#####################################################################################################
#####################################################################################################
#####################################################################################################
authors_list = dl_obj.authors
tfe_obj = TextFeaturesExtractor(input_data=None)
authors_graph = tfe_obj.create_authors_graph(authors=authors_list)
gfe_obj = GraphFeaturesExtractor(graph=authors_graph)

authors_embeddings_index = gfe_obj.create_node2vec_embeddings(
    emb_size=NODE_2_VEC_EMB_SIZE,
    filename='glove.authors.graph.nodes',
    load_embeddings=True)

cleaned_authors = [tfe_obj.clean_up_authors(authors) for authors in authors_list]

all_authors = list(filter(None, {item for sublist in cleaned_authors for item in sublist}))

word_index = {author: i + 1 for i, author in enumerate(all_authors + ['<UNK>'])}

print('Found {} unique authors tokens.'.format(len(word_index)))

authors_input_size = len(authors_embeddings_index) + 1
# constructing an embedding matrix
authors_embedding_matrix = np.random.random((authors_input_size, NODE_2_VEC_EMB_SIZE))
for word, i in word_index.items():
    author_embedding_vector = authors_embeddings_index.get(word)
    if author_embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        authors_embedding_matrix[i] = author_embedding_vector

print('Authors embedding matrix shape: ', authors_embedding_matrix.shape)

x_train_authors = dl_obj.x_train['authors']
x_val_authors = dl_obj.x_val['authors']
x_test_authors = dl_obj.x_test['authors']

x_train_authors_list = [tfe_obj.clean_up_authors(authors) for authors in x_train_authors]
x_val_authors_list = [tfe_obj.clean_up_authors(authors) for authors in x_val_authors]
x_test_authors_list = [tfe_obj.clean_up_authors(authors) for authors in x_test_authors]

authors_max_length = 0
x_train_sequences = list()
for authors_lst in x_train_authors_list:
    authors_seqs = [word_index.get(author, word_index.get('<UNK>')) for author in authors_lst]
    authors_max_length = len(authors_seqs) if len(authors_seqs) > authors_max_length else authors_max_length
    x_train_sequences.append(authors_seqs)

x_val_sequences = list()
for authors_lst in x_val_authors_list:
    authors_seqs = [word_index.get(author, word_index.get('<UNK>')) for author in authors_lst]
    x_val_sequences.append(authors_seqs)

x_test_sequences = list()
for authors_lst in x_test_authors_list:
    authors_seqs = [word_index.get(author, word_index.get('<UNK>')) for author in authors_lst]
    x_test_sequences.append(authors_seqs)

x_train_authors_padded = pad_sequences(x_train_sequences, maxlen=authors_max_length, padding='post')
x_val_authors_padded = pad_sequences(x_val_sequences, maxlen=authors_max_length, padding='post')
x_test_authors_padded = pad_sequences(x_test_sequences, maxlen=authors_max_length, padding='post')


############################################################################
############################################################################
############################################################################

# define model

def build_model(rnn_size,
                rnn_emb_size,
                abstracts_max_length,
                abstracts_voc_size,
                titles_max_length,
                titles_voc_size,
                authors_max_length,
                authors_input_size,
                authors_embedding_matrix,
                node_2_vec_emb_size,
                regularization,
                static_input_size,
                dropout,
                opt):
    """

    :return:
    """
    # abstract part
    abstract_input = Input(shape=(abstracts_max_length,),
                           dtype='int32',
                           name='abstract_input')
    embedded_abstract_text = layers.Embedding(abstracts_voc_size,
                                              rnn_emb_size,
                                              name='abstract_emb_layer')(abstract_input)
    encoded_abstract_text = layers.LSTM(rnn_size,
                                        dropout=dropout,
                                        recurrent_dropout=dropout,
                                        name='abstract_output_layer')(embedded_abstract_text)

    # title part
    title_input = Input(shape=(titles_max_length,),
                        dtype='int32',
                        name='title_input')

    embedded_title_text = layers.Embedding(titles_voc_size,
                                           rnn_emb_size,
                                           name='title_emb_layer')(title_input)

    encoded_title_text = layers.LSTM(rnn_size,
                                     dropout=dropout,
                                     recurrent_dropout=dropout,
                                     name='title_output_layer')(embedded_title_text)

    # authors part
    authors_input = layers.Input(shape=(authors_max_length,),
                                 dtype='int32',
                                 name='authors_input')
    authors_emb_layer = layers.Embedding(authors_input_size,
                                         node_2_vec_emb_size,
                                         weights=[authors_embedding_matrix],
                                         input_length=authors_max_length,
                                         trainable=True,
                                         name='authors_emb_layer')

    embedded_sequences = authors_emb_layer(authors_input)
    encoded_authors = layers.Bidirectional(layers.LSTM(rnn_size,
                                                       dropout=dropout,
                                                       recurrent_dropout=dropout),
                                           name='authors_output_layer')(embedded_sequences)

    # metrics part
    metrics_input = Input(shape=(static_input_size,),
                          dtype='float32',
                          name='metrics_input')
    metrics_deep_1 = layers.Dense(128,
                                  activation='tanh',
                                  kernel_initializer='glorot_normal',
                                  kernel_regularizer=regularization,
                                  name='metrics_deep_layer_1')(metrics_input)
    metrics_deep_1 = layers.BatchNormalization(name='metrics_deep_layer_1_batch_norm')(metrics_deep_1)
    metrics_deep_1 = layers.Dropout(dropout, name='metrics_deep_layer_1_dropout')(metrics_deep_1)

    # concatenation of the 4 layers
    merged_layer = layers.concatenate([encoded_abstract_text,
                                       encoded_title_text,
                                       encoded_authors,
                                       metrics_deep_1],
                                      axis=-1,
                                      name='merged_layer')

    merged_layer_1 = layers.Dense(64,
                                  activation='tanh',
                                  kernel_initializer='glorot_normal',
                                  kernel_regularizer=regularization,
                                  name='merged_deep_layer_1')(merged_layer)

    merged_layer_1 = layers.BatchNormalization(name='merged_deep_layer_1_batch_norm')(merged_layer_1)
    merged_layer_1 = layers.Dropout(dropout,
                                    name='merged_deep_layer_1_dropout')(merged_layer_1)

    category = layers.Dense(28, activation='softmax', name='main_output')(merged_layer_1)

    mixed_model = Model([abstract_input,
                         title_input,
                         authors_input,
                         metrics_input],
                        category)

    mixed_model.compile(optimizer=opt,
                        loss='categorical_crossentropy',
                        metrics=['acc'])

    print(mixed_model.summary())
    return mixed_model


DROPOUT = 0.7
RNN_EMB_SIZE = 300
RNN_SIZE = 100
STATIC_INPUT_SIZE = x_train_static.shape[1]
lr = 0.001
regularization = regularizers.l2(0.01)
N_EPOCHS = 100
BATCH_SIZE = 128
model_outfile = os.path.join(MODELS_DIR, 'all_inputs_model.h5')

abstracts_voc_size = len(x_train_abstracts_int2word)
titles_voc_size = len(x_train_titles_int2word)

opt = Adam(lr=lr, decay=0.0)
# opt = RMSprop(lr=lr, decay=0.0)

model = build_model(rnn_size=RNN_SIZE,
                    rnn_emb_size=RNN_EMB_SIZE,
                    abstracts_max_length=x_train_abstracts_max_length,
                    abstracts_voc_size=abstracts_voc_size,
                    titles_max_length=x_train_titles_max_length,
                    titles_voc_size=titles_voc_size,
                    authors_max_length=authors_max_length,
                    authors_input_size=authors_input_size,
                    authors_embedding_matrix=authors_embedding_matrix,
                    node_2_vec_emb_size=NODE_2_VEC_EMB_SIZE,
                    regularization=regularization,
                    static_input_size=STATIC_INPUT_SIZE,
                    dropout=DROPOUT,
                    opt=opt)

callbacks_list = [
    callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS_DIR,
                          histogram_freq=0,
                          embeddings_freq=1,
                          write_graph=True,
                          write_images=False),
    callbacks.EarlyStopping(monitor='acc',
                            patience=10),
    callbacks.ModelCheckpoint(filepath=model_outfile,
                              monitor='val_loss',
                              save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss',
                                factor=0.1,
                                patience=2)]

x_train_input = [x_train_abstracts_padded,
                 x_train_titles_padded,
                 x_train_authors_padded,
                 x_train_static]

x_val_input = [x_val_abstracts_padded,
               x_val_titles_padded,
               x_val_authors_padded,
               x_val_static]

history = model.fit(x=x_train_input,
                    y=y_train_one_hot,
                    epochs=N_EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_val_input, y_val_one_hot),
                    verbose=2,
                    callbacks=callbacks_list)

from rpcc.evaluation import plot_model_metadata

plot_model_metadata(history)
