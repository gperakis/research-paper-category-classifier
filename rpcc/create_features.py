import itertools
import re
from itertools import combinations
from random import choice

import networkx as nx
import pandas as pd
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from more_itertools import windowed, flatten
from networkx import DiGraph
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
    graph: DiGraph

    def __init__(self, graph):
        """

        :param graph:
        """
        self.graph = graph

    def random_walk(self,
                    node: int,
                    walk_length: int) -> list:
        """
        This function performs a random walk of size "walk_length" for a given "node" from a directed graph G.
        :param graph: A network X directed graph
        :param node: One of the nodes of the graph
        :param walk_length:
        :return:
        """

        assert node in self.graph.nodes()
        walk = [node]

        for i in range(walk_length):
            # create a list of the successors for the given node
            successors = list(self.graph.successors(walk[-1]))
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

        for node in self.graph.nodes():
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

        :param graph: A networkX graph
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

        for node in self.graph.nodes():
            embeddings[node] = model[node]

        return embeddings

    @property
    def calculate_avg_neighbour_degree(self) -> dict:
        """

        :return:
        """
        return nx.average_neighbor_degree(self.graph)

    @property
    def calculate_out_degree(self) -> dict:
        """

        :return:
        """
        return {t[0]: t[1] for t in self.graph.out_degree}

    @property
    def calculate_in_degree(self) -> dict:
        """

        :return:
        """
        return {t[0]: t[1] for t in self.graph.in_degree}

    @property
    def calculate_undirected_degree(self) -> dict:
        """

        :return:
        """
        return {t[0]: t[1] for t in self.graph.to_undirected().degree}

    @property
    def calculate_out_degree_centrality(self) -> dict:
        """

        :return:
        """
        return nx.out_degree_centrality(self.graph)

    @property
    def calculate_in_degree_centrality(self) -> dict:
        """

        :return:
        """
        return nx.in_degree_centrality(self.graph)

    @property
    def calculate_undirected_degree_centrality(self) -> dict:
        """

        :return:
        """
        return nx.degree_centrality(self.graph.to_undirected())

    @property
    def calculate_betweenness_centrality(self) -> dict:
        """

        :return:
        """
        return nx.betweenness_centrality(self.graph)

    @property
    def calculate_closeness_centrality(self) -> dict:
        """

        :return:
        """
        return nx.closeness_centrality(self.graph)

    @property
    def calculate_page_rank(self) -> dict:
        """

        :return:
        """
        return nx.pagerank(self.graph, alpha=0.9)

    @property
    def calculate_hub_and_authorities(self) -> tuple:
        """

        :return:
        """
        hubs, authorities = nx.hits(self.graph)
        return hubs, authorities

    @property
    def calculate_number_of_triangles(self) -> dict:
        """

        :return:
        """
        return nx.triangles(self.graph.to_undirected())

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

    directed_graph = TextFeaturesExtractor.generate_graph_from_text(text=abstract, stop_word=None)

    obj = GraphFeaturesExtractor(directed_graph)

    print(obj.calculate_avg_neighbour_degree)
    print(obj.calculate_out_degree)
    print(obj.calculate_in_degree)
    print(obj.calculate_undirected_degree)
    print(obj.calculate_out_degree_centrality)
    print(obj.calculate_in_degree_centrality)
    print(obj.calculate_undirected_degree_centrality)
    print(obj.calculate_betweenness_centrality)
    print(obj.calculate_closeness_centrality)
    print(obj.calculate_page_rank)
    print(obj.calculate_hub_and_authorities)
    print(obj.calculate_number_of_triangles)
