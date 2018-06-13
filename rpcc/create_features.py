import itertools
import re
from itertools import combinations

import networkx as nx
import pandas as pd
from more_itertools import windowed, flatten
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

    def __init__(self,
                 input_data):
        """

        :param input_data:
        """

        self.raw_data = input_data
        self.processed_data = None

        self._create_features()

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
