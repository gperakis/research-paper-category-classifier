import itertools
import os
import pickle
import re
from itertools import combinations
from random import choice
from rpcc import load_data
import networkx as nx
import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical
from more_itertools import windowed, flatten
from networkx.algorithms.community.label_propagation import label_propagation_communities
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
SPACY_NLP = spacy.load('en', parse=False, tag=False, entity=False)

from rpcc import PROCESSED_DATA_DIR, CONTRACTION_MAP, RAW_DATA_DIR


class FeatureExtractor:
    """
    This class should be initialized by a given dataset in Pandas DataFrame format,
    in which could be performed various feature creations.

    The expected input DataFrame should have at least the following columns:
        - paper_title
        - paper_authors
        - paper_abstract

    The expected produced Pandas DataFrame should have the following columns which
    will serve as the final dataset to be feed to the model:
        - title_embedding
        - abstract_embedding
        - author_embedding
        - paper_community
        - paper_centrality_metrics

    """

    def __init__(self, input_data=None):
        """

        :param input_data:
        """

        self.raw_data = input_data
        self.processed_data = None

        self._create_features()

    def _create_features(self):
        """
        This method performs the transformations in self.input_data in order to produce
        the final dataset.
        It runs all the feature transformations one by one and it assembles the outputs
        to one Pandas DataFrame

        :return: Pandas DataFrame, is stored in object's process_data variable.
        """

        # TODO run one by one each feature creation

        self.processed_data = self.raw_data


class TextFeaturesExtractor(FeatureExtractor):

    def __init__(self, input_data):
        """

        :param input_data:
        """
        super().__init__(input_data)

    @staticmethod
    def clean_up_authors(authors):
        """
        Extract author names from a string

        :param authors: str. the authors fo the article
        :return: list of str. with the authors of the article, is applicable
        """
        if authors is None:
            return []

        else:
            temp_authors = re.sub(r'\((.*?)\)', '', authors)
            temp_authors = re.sub(r'\((.*?,)', '', temp_authors)
            temp_authors = re.sub(r'\((.*?)', '', temp_authors)

            cleaned_authors = temp_authors.split(',')
            cleaned_authors = [a.strip().lower() for a in cleaned_authors]
            cleaned_authors = [author for author in cleaned_authors if len(author) > 2]

            return cleaned_authors

    @staticmethod
    def expand_contractions(text: str) -> str:
        """
        This function expands contractions for the english language. For example "I've" will become "I have".

        :param text:
        :return:
        """

        contractions_pattern = re.compile('({})'.format(
            '|'.join(CONTRACTION_MAP.keys())), flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            """
            This sub function helps into expanding a given contraction
            :param contraction:
            :return:
            """
            match = contraction.group(0)
            first_char = match[0]

            expanded = CONTRACTION_MAP.get(match) if CONTRACTION_MAP.get(match) else CONTRACTION_MAP.get(match.lower())

            expanded = first_char + expanded[1:]

            return expanded

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)

        return expanded_text

    @staticmethod
    def lemmatize_text(text, spacy_nlp=SPACY_NLP):
        """
        This method lemmatizes text

        :param text:
        :return:
        """
        text = spacy_nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

        return text

    def create_authors_graph(self, authors: iter) -> nx.Graph:
        """
        Creates a networkX undirected network object with weighted edges that
        stores the connections between the authors

        :param authors: An iterable of strings containing multiple authors in each string.
        :return: A networkX graph of all the connections between the authors.
        """
        if authors:

            # instantiating a simple Graph.
            G = nx.DiGraph()

            for authors_str in authors:
                # cleaning up the authors. Returns a list of authors.
                cleaned_authors = self.clean_up_authors(authors_str)
                # only keeping those authors that have length over 2 characters.

                if len(cleaned_authors) > 1:
                    # extracting all author combinations per pair.
                    for comb in itertools.combinations(cleaned_authors, 2):
                        # if there is already an edge between the two authors, add more weight.
                        if G.has_edge(comb[0], comb[1]):
                            G[comb[0]][comb[1]]['weight'] += 1
                        else:
                            G.add_edge(comb[0], comb[1], weight=1)
                elif len(cleaned_authors) == 1:
                    G.add_node(cleaned_authors[0])

            return G

        else:
            raise NotImplementedError('Must load Node INFO first.')

    def pre_process_text(self, texts: iter, remove_stopwords: bool = False) -> dict:
        """
        This method instantiates a tokenizer, and fits that tokenizer with the texts.
        Then creates tokenized sequences, calculates maximum texts length and padd the shorter texts

        :param texts: An iterable of strings
        :return: dict. A dictionary containing metadata that are userfull when building DL models.
        """
        print('Setting Tokenizer')
        # setting up word level tokenizer
        tokenizer = Tokenizer(char_level=False, oov_token='<UNK>')

        texts_clean = list()
        print('Run Expand Contractions')
        # removing whitespaces, converting to lower case
        for text in texts:
            text = text.lower().strip()
            expanded = self.expand_contractions(text=text)
            # lemmatized = self.lemmatize_text(text=expanded)
            if remove_stopwords:
                expanded = ' '.join([i for i in expanded.lower().split() if i not in STOPWORDS])
            texts_clean.append(expanded)

        print('Fitting Tokenizer')
        # creating the vocabulary of the tokenizer
        tokenizer.fit_on_texts(texts=texts_clean)
        # converting in sequences of integers.
        print('Transforming Text to Integer Sequences')
        tokenized_sequences = tokenizer.texts_to_sequences(texts_clean)

        print('Computing max length')
        # calculating the max_length.
        max_length = max([len(seq) for seq in tokenized_sequences])

        # padding the shorter sentences by adding zeros
        print('Padding Sequences')
        padded_sequences = pad_sequences(tokenized_sequences,
                                         maxlen=max_length,
                                         dtype='int32',
                                         padding='post',
                                         truncating='post')

        # creating 2 dictionaries that will help in the dl processes
        print('Creating Glossary Dicts')
        int2word = {num: char for char, num in tokenizer.word_index.items()}
        word2int = tokenizer.word_index

        return dict(x=padded_sequences,
                    int2word=int2word,
                    word2int=word2int,
                    max_length=max_length,
                    tokenizer=tokenizer)

    def text_to_padded_sequences(self,
                                 texts: iter,
                                 tokenizer: Tokenizer,
                                 max_length: int,
                                 remove_stopwords: bool = False) -> list:
        """

        :param texts:
        :param tokenizer:
        :param max_length:
        :param remove_stopwords:

        :return:
        """

        texts_clean = list()

        for text in texts:
            text = text.lower().strip()
            expanded = self.expand_contractions(text=text)
            # lemmatized = self.lemmatize_text(text=expanded)
            if remove_stopwords:
                expanded = ' '.join([i for i in expanded.lower().split() if i not in STOPWORDS])
            texts_clean.append(expanded)

        # converting in sequences of integers.
        tokenized_sequences = tokenizer.texts_to_sequences(texts_clean)
        # padding the shorter sentences by adding zeros
        padded_sequences = pad_sequences(tokenized_sequences,
                                         maxlen=max_length,
                                         dtype='int32',
                                         padding='post',
                                         truncating='post')

        return padded_sequences

    def generate_graph_from_text(self,
                                 text: str,
                                 window_size: int = 3,
                                 directed: bool = True) -> nx.Graph:
        """
        This method creates a graph from the words of a text. At first splits the text in sentences
        and then parses each sentence through a  sliding window, generating a graph by joining the window tokens with
        some weights. This text may be an abstract of a paper or it's title.

        :type text: str
        :type window_size: int
        :type directed: bool
        :param text: A document of any size.
        :param window_size: The size of the sliding window that we will use in order to generate the nodes and edges.
        :param directed: Whether we will created a directed or undirected graph.
        :return: A networkX graph.
        """
        assert directed in [True, False]
        assert window_size in range(1, 6)

        expanded = self.expand_contractions(text=text)
        # lemmatized = self.lemmatize_text(text=expanded)

        tokens = text_to_word_sequence(text=expanded, lower=True)

        if directed:
            # instantiating the Directed graph
            G = nx.DiGraph()
        else:
            # instantiating the Undirected graph
            G = nx.Graph()

        # All the magic is here!  Creates the weights indices.
        # For example for window_size = 5 the output is weight_index = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1]
        # The weight index remain the same for all windows due to the fact that the window size is stable.
        weight_index = list(flatten([list(range(1, i + 1)) for i in range(window_size - 1, 0, -1)]))

        for window in windowed(tokens, window_size):
            # for the tokens in the window take all the combinations.
            # Eg: Tokens: [tok1, tok2, tok3, tok4] the result will be:
            # [tok1, tok2], [tok1, tok3], [tok1, tok4], [tok2, tok3], [tok2, tok4], [tok3, tok4]
            for num, comb in enumerate(combinations(window, 2)):
                # create the actual weight for each combination: eg: 1, 1/2, 1/3, 1/4, etc
                weight = 1 / weight_index[num]

                # if there is already an edge between the two text tokens, add more weight between the edges.
                if G.has_edge(comb[0], comb[1]):
                    G[comb[0]][comb[1]]['weight'] += weight
                else:
                    # there is no edge, so we create an edge between them.
                    G.add_edge(comb[0], comb[1], weight=weight)

        # removing any self loop edges.
        G.remove_edges_from(G.selfloop_edges())

        return G


class GraphFeaturesExtractor:

    def __init__(self, graph):
        """

        :param graph:
        """
        self.directed_graph = graph

        self.undirected_graph = self.directed_graph.to_undirected()

    def random_walk(self,
                    node: int,
                    walk_length: int) -> list:
        """
        This function performs a random walk of size "walk_length" for a given "node" from a directed graph G.
        :param node: One of the nodes of the graph
        :param walk_length:
        :return:
        """

        assert node in self.directed_graph.nodes()
        walk = [node]

        for i in range(walk_length):
            # create a list of the successors for the given node
            successors = list(self.directed_graph.successors(walk[-1]))
            if successors:
                walk.append(choice(successors))
            else:
                # if no successors stop
                break

        return walk

    def generate_walks(self,
                       num_walks: int,
                       walk_length: int) -> list:
        """
        This method for a given graph creates a number of "num_walks" for all nodes of the given walk length

        :param num_walks: Int. The number of walks to be extracted for each node.
        :param walk_length: Int. The size of the of walk length
        :return:
        """
        walks = list()

        for node in self.directed_graph.nodes():
            for n_walk in range(num_walks):
                walk_list = self.random_walk(node=node, walk_length=walk_length)
                walks.append(walk_list)

        return walks

    def learn_embeddings(self,
                         walks: iter,
                         window_size: int,
                         d: int):
        """
        This method creates node embeddings given a graph and some random 'walks'.

        :param walks: An list of lists of integers denoting the nodes of the graph.
        :param window_size: The size of the windows of the model that learns the embeddings.
        :param d: The output dimension of the node embeddings.
        :return:
        """
        model = Word2Vec(sentences=walks,
                         size=d,
                         min_count=0,
                         window=window_size,
                         iter=100,
                         workers=-1,
                         sg=1)

        embeddings = dict()

        for node in self.directed_graph.nodes():
            embeddings[node] = model[node]

        return embeddings

    @property
    def calculate_avg_neighbour_degree(self) -> dict:
        """

        :return:
        """
        return nx.average_neighbor_degree(self.directed_graph)

    @property
    def calculate_out_degree(self) -> dict:
        """

        :return:
        """
        return {t[0]: t[1] for t in self.directed_graph.out_degree}

    @property
    def calculate_in_degree(self) -> dict:
        """

        :return:
        """
        return {t[0]: t[1] for t in self.directed_graph.in_degree}

    @property
    def calculate_undirected_degree(self) -> dict:
        """

        :return:
        """
        return {t[0]: t[1] for t in self.undirected_graph.degree}

    @property
    def calculate_out_degree_centrality(self) -> dict:
        """

        :return:
        """
        return nx.out_degree_centrality(self.directed_graph)

    @property
    def calculate_in_degree_centrality(self) -> dict:
        """

        :return:
        """
        return nx.in_degree_centrality(self.directed_graph)

    @property
    def calculate_undirected_degree_centrality(self) -> dict:
        """

        :return:
        """
        return nx.degree_centrality(self.undirected_graph)

    @property
    def calculate_betweenness_centrality(self) -> dict:
        """

        :return:
        """
        return nx.betweenness_centrality(self.directed_graph)

    @property
    def calculate_closeness_centrality(self) -> dict:
        """

        :return:
        """
        return nx.closeness_centrality(self.directed_graph)

    @property
    def calculate_page_rank(self) -> dict:
        """

        :return:
        """
        return nx.pagerank(self.directed_graph, alpha=0.9)

    @property
    def calculate_hub_and_authorities(self) -> tuple:
        """

        :return:
        """
        hubs, authorities = nx.hits(self.directed_graph)
        return hubs, authorities

    @property
    def calculate_number_of_triangles(self) -> dict:
        """

        :return:
        """
        return nx.triangles(self.undirected_graph)

    @property
    def extract_k_core_nodes(self) -> list:
        """
        This function extracts the nodes of the k-core sub-graph of a given graph.

        :return:
        """
        core = nx.k_core(self.directed_graph)
        return core.nodes()

    @property
    def convert_graph_to_adjacency_matrix(self) -> np.matrix:
        """

        :return:
        """
        return nx.to_numpy_matrix(self.directed_graph)

    def get_node_cliques_metrics(self, node) -> dict:
        """

        :param node:
        :return:
        """

        out = {'node_cliques_size_avg': 0,
               'node_cliques_size_std': 0,
               'node_cliques_size_max': 0,
               'node_number_of_cliques': 0}

        cliques = nx.cliques_containing_node(self.undirected_graph, nodes=node)

        if cliques:
            clique_sizes = [len(c) for c in cliques]

            out['node_cliques_size_avg'] = np.mean(clique_sizes)
            out['node_cliques_size_std'] = np.std(clique_sizes)
            out['node_cliques_size_max'] = max(clique_sizes)
            out['node_number_of_cliques'] = len(cliques)

        return out

    @property
    def get_graph_cliques_metrics(self) -> tuple:
        """

        :return:
        """
        cliques_size_avg_dict = dict()
        cliques_size_std_dict = dict()
        cliques_size_max_dict = dict()
        number_of_cliques_dict = dict()

        for node in self.directed_graph.nodes():
            meta = self.get_node_cliques_metrics(node)
            cliques_size_avg_dict[node] = meta['node_cliques_size_avg']
            cliques_size_std_dict[node] = meta['node_cliques_size_std']
            cliques_size_max_dict[node] = meta['node_cliques_size_max']
            number_of_cliques_dict[node] = meta['node_number_of_cliques']

        return cliques_size_avg_dict, cliques_size_std_dict, cliques_size_max_dict, number_of_cliques_dict

    @property
    def get_one_hot_communities(self) -> pd.DataFrame:
        """

        :return:
        """
        community_generator = label_propagation_communities(self.undirected_graph)

        number_of_communities = 0
        nodes = list()
        community_labels = list()
        for community_id, community_members in enumerate(community_generator):
            number_of_communities += 1
            for node in community_members:
                nodes.append(node)
                community_labels.append(community_id)

        one_hot_communities = to_categorical(y=community_labels,
                                             num_classes=number_of_communities)

        communities_df = pd.DataFrame(one_hot_communities,
                                      index=nodes,
                                      columns=["comm_{}".format(i) for i in range(number_of_communities)])

        return communities_df

    def create_full_node_features(self) -> pd.DataFrame:
        """
        This method performs the transformations in self.graph in order to produce the final dataset.
        It runs all the feature transformations one by one and it assembles the outputs
        to one Pandas DataFrame

        :return: Pandas DataFrame, is stored in object's process_data variable.
        """
        avg_neigh_degree_dict = self.calculate_avg_neighbour_degree
        out_degree_dict = self.calculate_out_degree
        in_degree_dict = self.calculate_in_degree
        degree_dict = self.calculate_undirected_degree
        out_degree_centrality_dict = self.calculate_out_degree_centrality
        in_degree_centrality_dict = self.calculate_in_degree_centrality
        degree_centrality_dict = self.calculate_undirected_degree_centrality
        betweenness_centrality_dict = self.calculate_betweenness_centrality
        closeness_centrality_dict = self.calculate_closeness_centrality
        page_rank_dict = self.calculate_page_rank
        hubs_dict, authorities_dict = self.calculate_hub_and_authorities
        n_triangles_dict = self.calculate_number_of_triangles

        # Metadata regarding cliques
        avg_size_dict, std_size_dict, max_size_dict, number_of_cliques_dict = self.get_graph_cliques_metrics

        node_metrics = list()
        for node in self.directed_graph.nodes():
            node_features = {
                'node': node,
                'avg_neigh_deg': avg_neigh_degree_dict[node],
                'out_degree': out_degree_dict[node],
                'in_degree': in_degree_dict[node],
                'degree': degree_dict[node],
                'out_degree_centrality': out_degree_centrality_dict[node],
                'in_degree_centrality': in_degree_centrality_dict[node],
                'degree_centrality': degree_centrality_dict[node],
                'betweenness_centrality': betweenness_centrality_dict[node],
                'closeness_centrality': closeness_centrality_dict[node],
                'page_rank': page_rank_dict[node],
                'hubs_rank': hubs_dict[node],
                'authorities_rank': authorities_dict[node],
                'n_triangles': n_triangles_dict[node],
                'avg_cliques_size': avg_size_dict[node],
                'std_cliques_size': std_size_dict[node],
                'max_clique_size': max_size_dict[node],
                'n_cliques': number_of_cliques_dict[node],

            }

            node_metrics.append(node_features)

        node_metrics_df = pd.DataFrame(node_metrics)
        return node_metrics_df

    def create_simple_node_features(self,
                                    load_metrics: bool = False,
                                    save_metrics: bool = True,
                                    outfile: str = 'graph_simple_metrics.csv') -> pd.DataFrame:
        """
        This method performs the transformations in self.graph in order to produce the final dataset.
        It runs all the feature transformations one by one and it assembles the outputs
        to one Pandas DataFrame

        :return: Pandas DataFrame, is stored in object's process_data variable.
        """
        if load_metrics:
            outfile = os.path.join(PROCESSED_DATA_DIR, outfile)
            return pd.read_csv(outfile)

        print('calculating avg neighbour degree')
        avg_neigh_degree_dict = self.calculate_avg_neighbour_degree
        print('calculating out degree')
        out_degree_dict = self.calculate_out_degree
        print('calculating in degree')
        in_degree_dict = self.calculate_in_degree
        print('calculating undirected degree')
        degree_dict = self.calculate_undirected_degree
        print('calculating out degree centrality')
        out_degree_centrality_dict = self.calculate_out_degree_centrality
        print('calculating in degree centrality')
        in_degree_centrality_dict = self.calculate_in_degree_centrality
        print('calculating undirected degree centrality')
        degree_centrality_dict = self.calculate_undirected_degree_centrality
        print('Done')

        node_metrics = list()
        for node in self.directed_graph.nodes():
            node_features = {
                'node': str(node),
                'avg_neigh_deg': avg_neigh_degree_dict[node],
                'out_degree': out_degree_dict[node],
                'in_degree': in_degree_dict[node],
                'degree': degree_dict[node],
                'out_degree_centrality': out_degree_centrality_dict[node],
                'in_degree_centrality': in_degree_centrality_dict[node],
                'degree_centrality': degree_centrality_dict[node],
            }

            node_metrics.append(node_features)

        node_metrics_df = pd.DataFrame(node_metrics)

        if save_metrics:
            outfile = os.path.join(PROCESSED_DATA_DIR, outfile)
            node_metrics_df.to_csv(outfile, index=False, encoding='utf-8')

        return node_metrics_df

    def create_node2vec_embeddings(self,
                                   emb_size=200,
                                   filename: str = 'glove.citation.graph.nodes',
                                   load_embeddings: bool = False,
                                   save_embeddings: bool = True
                                   ):
        """

        :param emb_size:
        :param filename:
        :param load_embeddings:
        :param save_embeddings:
        :return:
        """
        assert emb_size in [50, 100, 200, 300]

        filename = '{}.{}d.pickle'.format(filename, emb_size)
        filepath = os.path.join(PROCESSED_DATA_DIR, 'embeddings', filename)

        if load_embeddings:
            print('Loading Embeddings file: {}'.format(filepath))
            with open(filepath, 'rb') as handle:
                embeddings_dict = pickle.load(handle)
                return embeddings_dict

        walks = self.generate_walks(num_walks=10, walk_length=10)

        unique_walks = [list(x) for x in set(tuple(x) for x in walks)]

        embeddings_dict = self.learn_embeddings(unique_walks, window_size=5, d=emb_size)

        if save_embeddings:
            print('Saving Embeddings file: {}'.format(filepath))
            with open(filepath, 'wb') as handle:
                pickle.dump(embeddings_dict,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        return embeddings_dict


def create_node2vec_embeddings_from_texts(texts: iter,
                                          window_size: int = 3,
                                          emb_size: int = 200,
                                          filename: str = 'glove.abstracts.nodes',
                                          load_embeddings: bool = False,
                                          save_embeddings: bool = True) -> dict:
    """
    This function creates a large corpus of text from an iterable of texts. Then tokenizes the text and
    creates a large directed graph.
    Then it takes all nodes from the graph, creates multiple random paths and then feeds this path to
    gensim in order to obtain Glove embeddings.



    :param texts: An iterable of texts.
    :param window_size: The size of the sliding window that will be used for the creation of the directed graph.
    :param emb_size: int. The size of the embedding GlOVE vectors.
    :param filename: str. The filename in order to load or save the node2vec embeddings.
    :param load_embeddings: bool. Whether we want to load a pre-trained embeddings model.
    :param save_embeddings: bool. Whether we want to save the trained embeddings model.
    :return: dict. A dictionary of words (nodes) to vec embeddings.
    """
    assert emb_size in [50, 100, 200, 300]

    filename = '{}.{}d.pickle'.format(filename, emb_size)
    filepath = os.path.join(PROCESSED_DATA_DIR, 'embeddings', filename)

    if load_embeddings:
        print('Loading Embeddings file: {}'.format(filepath))
        with open(filepath, 'rb') as handle:
            embeddings_dict = pickle.load(handle)
            return embeddings_dict

    corpus = '\n'.join(texts)

    # contractions expanding and lemmatization is done within 'generate_graph_from_text'
    print('Creating directed Graph')
    directed_graph = TextFeaturesExtractor(input_data=None).generate_graph_from_text(text=corpus,
                                                                                     window_size=window_size)

    graph_obj = GraphFeaturesExtractor(graph=directed_graph)
    print('Creating Embeddings of size: {}'.format(emb_size))
    embeddings_dict = graph_obj.create_node2vec_embeddings(emb_size=emb_size)

    if save_embeddings:
        print('Saving Embeddings file: {}'.format(filepath))
        with open(filepath, 'wb') as handle:
            pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return embeddings_dict


if __name__ == "__main__":
    pass
    #################################################################################
    # papers_path = os.path.join(RAW_DATA_DIR, 'node_information.csv')
    #
    # # abstracts =pd.read_csv(papers_path, usecols=['abstract'])['abstract']
    # titles = pd.read_csv(papers_path, usecols=['title'])['title']
    #
    # for emb_size in [50, 100, 200, 300]:
    #     create_node2vec_embeddings_from_texts(texts=titles,
    #                                           window_size=3,
    #                                           emb_size=emb_size,
    #                                           filename='glove.titles.nodes',
    #                                           load_embeddings=False,
    #                                           save_embeddings=True)
    #################################################################################

    #################################################################################
    # dl_obj = load_data.DataLoader()
    # citations_graph = dl_obj.create_citation_network()
    # gfe_obj = GraphFeaturesExtractor(graph=citations_graph)
    # for emb_d in [50, 100, 200, 300]:
    #     gfe_obj.create_node2vec_embeddings(emb_size=emb_d,
    #                                        filename='glove.citation.graph.nodes',
    #                                        save_embeddings=True,
    #                                        load_embeddings=False)
    #################################################################################
    #
    ################################################################################
    dl_obj = load_data.DataLoader()
    dl_obj.load_article_metadata()
    authors_list = dl_obj.authors
    tfe_obj = TextFeaturesExtractor(input_data=None)
    authors_graph = tfe_obj.create_authors_graph(authors=authors_list)
    gfe_obj = GraphFeaturesExtractor(graph=authors_graph)
    for emb_d in [50, 100, 200, 300]:
        gfe_obj.create_node2vec_embeddings(emb_size=emb_d,
                                           filename='glove.authors.graph.nodes',
                                           save_embeddings=True,
                                           load_embeddings=False)
    ################################################################################
