import itertools
import re
from itertools import combinations
from random import choice

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from more_itertools import windowed, flatten
from networkx import DiGraph
from networkx.algorithms.community.label_propagation import label_propagation_communities
from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer


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

    def __init__(self, input_data):
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
            cleaned_authors = [a.strip() for a in cleaned_authors]

            return cleaned_authors

    def __create_authors_graph(self, authors: iter) -> nx.Graph:
        """
        Creates a networkX undirected network object with weighted edges that
        stores the connections between the authors

        :param authors: An iterable of strings containing multiple authors in each string.
        :return: A networkX graph of all the connections between the authors.
        """
        if isinstance(authors, pd.Series):

            # instantiating a simple Graph.
            G = nx.Graph()

            for authors_str in authors:
                # cleaning up the authors. Returns a list of authors.
                cleaned_authors = self.clean_up_authors(authors_str)
                # only keeping those authors that have length over 2 characters.
                co_authors = [author for author in cleaned_authors if len(author) > 2]

                if co_authors:
                    # extracting all author combinations per pair.
                    for comb in itertools.combinations(co_authors, 2):

                        # if there is already an edge between the two authors, add more weight.
                        if G.has_edge(comb[0], comb[1]):
                            G[comb[0]][comb[1]]['weight'] += 1
                        else:
                            G.add_edge(comb[0], comb[1], weight=1)

            return G

        else:
            raise NotImplementedError('Must load Node INFO first.')

    @staticmethod
    def pre_process_text(texts: iter) -> dict:
        """
        This method istantiates a tokenizer, and fits that tokenizer with the texts.
        Then creates tokenized sequences, calculates maximum texts length and padd the shorter texts

        :param texts: An iterable of strings
        :return: dict. A dictionary containing metadata that are userfull when building DL models.
        """

        # setting up word level tokenizer
        tokenizer = Tokenizer(char_level=False, oov_token='<UNK>')

        texts_clean = list()
        # removing whitespaces, converting to lower case
        for text in texts:
            text = text.lower().strip()
            texts_clean.append(text)

        # creating the vocabulary of the tokenizer
        tokenizer.fit_on_texts(texts=texts_clean)
        # converting in sequences of integers.
        tokenized_sequences = tokenizer.texts_to_sequences(texts_clean)

        # calculating the max_length.
        max_length = max([len(seq) for seq in tokenized_sequences])

        # padding the shorter sentences by adding zeros
        padded_sequences = pad_sequences(tokenized_sequences,
                                         maxlen=max_length,
                                         dtype='int32',
                                         padding='post',
                                         truncating='post')

        # creating 2 dictionaries that will help in the dl processes
        int2word = {num: char for char, num in tokenizer.word_index.items()}
        word2int = tokenizer.word_index

        return dict(x=padded_sequences,
                    int2word=int2word,
                    word2int=word2int,
                    max_length=max_length,
                    tokenizer=tokenizer)

    @staticmethod
    def text_to_padded_sequences(texts: iter,
                                 tokenizer: Tokenizer,
                                 max_length: int) -> list:
        """

        :param texts:
        :param tokenizer:
        :param max_length:
        :return:
        """

        texts_clean = list()

        for text in texts:
            text = text.lower().strip()
            texts_clean.append(text)
        # converting in sequences of integers.
        tokenized_sequences = tokenizer.texts_to_sequences(texts_clean)
        # padding the shorter sentences by adding zeros
        padded_sequences = pad_sequences(tokenized_sequences,
                                         maxlen=max_length,
                                         dtype='int32',
                                         padding='post',
                                         truncating='post')

        return padded_sequences

    @staticmethod
    def generate_graph_from_text(text: str,
                                 window_size: int = 3,
                                 directed: bool = True,
                                 stop_word: str = 'english'):
        """
        This method creates a graph from the words of a text. At first splits the text in sentences
        and then parses each sentence through a  sliding window, generating a graph by joining the window tokens with
        some weights. This text may be an abstact of a paper or it's title.

        :type text: str
        :type window_size: int
        :type directed: bool
        :type stop_word: str
        :param text: A document of any size.
        :param window_size: The size of the sliding window that we will use in order to generate the nodes and edges.
        :param directed: Whether we will created a directed or undirected graph.
        :param stop_word: Stopwords for stop word removal.
        :return: A networkX graph.
        """
        assert directed in [True, False]
        assert window_size in range(1, 6)

        # splitting the text into sentences.
        sentences = sent_tokenize(text)

        # instantiating a tokenizer in order to tokenize each sentence.
        sent2tokens_tokenizer = CountVectorizer(stop_words=stop_word).build_analyzer()

        if directed:
            # instantiating the Directed graph
            G = nx.DiGraph()
        else:
            # instantiating the Undirted graph
            G = nx.Graph()

        # for each sentence split in tokens
        for sentence in sentences:
            tokens = sent2tokens_tokenizer(sentence)

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
    directed_graph: DiGraph

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
                         iter=50,
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

        :param graph:
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

        :param G:
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

    def _create_features(self) -> pd.DataFrame:
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


if __name__ == "__main__":
    # train_texts = ['This is a text',
    #                'This is another texts',
    #                'This is a third text that is very usefull']
    #
    # df = pd.DataFrame(train_texts, columns=['train_abstracts'])
    #
    # meta = TextFeaturesExtractor.pre_process_text(texts=df['train_abstracts'])
    #
    # x_train_padded = meta['x']
    # x_test_padded = TextFeaturesExtractor.text_to_padded_sequences(texts=df['train_abstracts'],
    #                                                                tokenizer=meta['tokenizer'],
    #                                                                max_length=meta['max_length'])
    # print(x_test_padded)

    abstract = 'This is one of the largest texts you will ever find largest'

    directed_graph = TextFeaturesExtractor.generate_graph_from_text(text=abstract,
                                                                    stop_word=None)

    obj = GraphFeaturesExtractor(directed_graph)

    asfd = obj.get_one_hot_communities

    print(asfd)
