import itertools
import os
import pickle
import re
from collections import Counter

import networkx as nx
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from rpcc import RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataLoader:

    def __init__(self,
                 train_file_name='train.csv',
                 test_file_name='test.csv',
                 graph_file_name='Cit-HepTh.txt',
                 info_file_name="node_information.csv",
                 verbose=0):
        """
        :param train_file_name: str. file name of the training data (features: article, journal)
        :param test_file_name: str. file name of the test data (features: article)
        :param graph_file_name: str. file name of the network structure
        :param info_file_name: str. file name of the data (features: article, title, year, authors, abstract)
        :param verbose: int. verbosity
        """

        self.train_file_name = os.path.join(RAW_DATA_DIR, train_file_name)
        self.test_file_name = os.path.join(RAW_DATA_DIR, test_file_name)
        self.graph_file_name = os.path.join(RAW_DATA_DIR, graph_file_name)
        self.info_file_name = os.path.join(RAW_DATA_DIR, info_file_name)

        self.verbose = verbose

        self.article_metadata = None

        # Graphs
        self.citation_graph = None
        self.authors_graph = None
        self.authors_label_props = None

        # X - features
        self.x_train_validation = None
        self.x_train = None
        self.x_val = None
        self.x_test = None

        # X - ids
        self.train_validation_ids = None
        self.train_ids = None
        self.validation_ids = None
        self.test_ids = None

        # y - targets
        self.y_train_validation = None
        self.y_train = None
        self.y_val = None

        # predictions
        self.y_predicted_test = None

        # Unique classes of the articles
        self.targets = None

    def get_stratified_data(self, X, y, val_size=0.2, random_state=0):
        """
        Performs stratified shuffle splitting to the given dataset

        :param X: pandas Dataframe with the data
        :param y: pandas Series with the targets
        :param val_size: float. percentage of the validation data set
        :param random_state: int. seed of the random shuffling
        :return: dict. with the splitted sets of data
        """
        sss = StratifiedShuffleSplit(test_size=val_size,
                                     random_state=random_state)

        X_train, X_val, y_train, y_val = None, None, None, None

        for train_index, val_index in sss.split(X, y):

            if self.verbose > 1:
                print("TRAIN:", train_index, "VALIDATION:", val_index)

            X_train, X_val = X.loc[train_index], X.loc[val_index]
            y_train, y_val = y.loc[train_index], y.loc[val_index]

        return {'x_train': X_train,
                'x_val': X_val,
                'y_train': y_train,
                'y_val': y_val}

    @staticmethod
    def __calc_label_ratios(x):
        """

        :param x:
        :return:
        """
        return sorted([(i, round(Counter(x)[i] / float(len(x)) * 100.0, 3)) for i in Counter(x)],
                      key=lambda x: x[1])

    def __load_article_metadata(self):
        """
        Reads data and stores it as Pandas DataFrame

        :return: Pandas DataFrame
        """
        # Load data about each article in a dataframe
        df = pd.read_csv(self.info_file_name)
        df['id'] = df['id'].apply(str)
        df = df.where((pd.notnull(df)), None)

        self.article_metadata = df

    def __create_citation_network(self):
        """
        Creates a networkX directed network object that stores the citations among the articles

        :return
        """
        # Create a directed graph
        G = nx.read_edgelist(self.graph_file_name,
                             delimiter='\t',
                             create_using=nx.DiGraph())

        if self.verbose > 0:
            print("Nodes: ", G.number_of_nodes())
            print("Edges: ", G.number_of_edges())

        self.citation_graph = G

    def __load_training_data(self):
        """
        Consolidates the labeled data with their metadata

        :return: pandas DataFrame with all the features and the target of both the train and validation data
        """
        train_val_df = pd.read_csv(self.train_file_name)
        train_val_df['Article'] = train_val_df['Article'].apply(str)
        train_val_df = train_val_df.where((pd.notnull(train_val_df)), None)

        train_val_enhanced = train_val_df.merge(self.article_metadata,
                                                left_on='Article',
                                                right_on='id',
                                                how='left')
        train_val_enhanced.drop('id', axis=1)

        self.train_validation_ids = list(train_val_df['Article'])
        self.x_train_validation = train_val_enhanced.drop('Journal', axis=1)
        self.y_train_validation = train_val_enhanced['Journal']

        self.targets = set(train_val_enhanced['Journal'])

        return train_val_enhanced

    def __load_test_data(self):
        """
        Consolidates the labeled data with their metadata and stores a pandas DataFrame
        with all the features of the test data

        """
        test_df = pd.read_csv(self.test_file_name)

        test_df['Article'] = test_df['Article'].apply(str)

        test_ids = list(test_df['Article'])

        test_x = self.article_metadata[self.article_metadata['id'].isin(test_ids)].copy()

        test_x.rename(columns={'id': 'Article'}, inplace=True)

        self.test_ids = test_ids
        self.x_test = test_x

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

    def __create_authors_label_props(self, train_val_enhanced):
        """
        Creates a dictionary with keys the authors and value each class with its probability

        :param train_val_enhanced: pandas DataFrame with all the data plus their target
        :return:
        """
        train_val_enhanced = train_val_enhanced[['Article', 'Journal', 'authors']]

        labels = train_val_enhanced['Journal'].unique().tolist()

        # Transform dataframe to a list of dictionaries list(dict(article, journal, authors))
        train_val_enhanced_recs = train_val_enhanced.to_dict('records')

        authors_labels_counts = list()
        for doc in train_val_enhanced_recs:
            cleaned_authors = self.clean_up_authors(doc['authors'])
            for author in cleaned_authors:
                if len(author) > 2:
                    authors_labels_counts.append((author, doc['Journal'], 1))

        df = pd.DataFrame(authors_labels_counts, columns=['author', 'Journal', 'value'])

        # Final dictionary with probs
        authors_subjects = dict()

        for author, author_df in df.groupby(['author']):
            all_labels_probs = dict.fromkeys(labels, 0.0)
            author_label_stats = (author_df.groupby('Journal').sum() / len(author_df)).to_dict().get('value', {})
            all_labels_probs.update(author_label_stats)

            authors_subjects[author] = all_labels_probs

        self.authors_label_props = authors_subjects

    def __create_authors_graph(self):
        """
        Creates a networkX undirected network object with weighted edges that stores the connections between the authors

        :return:
        """
        if isinstance(self.article_metadata, pd.DataFrame):

            if self.x_test is None:
                self.__load_test_data()

            if self.x_train is None:
                self.__load_training_data()

            # extracting all authors name for all papers in training validation and test data.
            train_val_test_authors = list(self.x_train_validation['authors']) + list(self.x_test['authors'])

            # instantiating a simple Graph.
            G = nx.Graph()

            for authors_str in train_val_test_authors:
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

            self.authors_graph = G
            return G

        else:
            raise NotImplementedError('Must load Node INFO first.')

    def run_data_preparation(self, val_size=0.2):
        """
        Creates all the needed datasets and networks for the given inputs.

        :param val_size: float. percentage of the validation data set
        :return:
        """

        # Load data about each article in a dataframe and set it to the constuctor
        self.__load_article_metadata()

        # Create a directed graph and store it in the constructor
        self.__create_citation_network()

        # Load consolidated train & validation dataframe
        train_val_enhanced = self.__load_training_data()

        # Load consolidated test dataframe
        self.__load_test_data()

        split_meta = self.get_stratified_data(X=self.x_train_validation,
                                              y=self.y_train_validation,
                                              val_size=val_size,
                                              random_state=0)

        # Stores splitting output into separate dictionaries
        self.x_train = split_meta['x_train']
        self.x_val = split_meta['x_val']
        self.y_train = split_meta['y_train']
        self.y_val = split_meta['y_val']

        self.train_ids = list(self.x_train['Article'])
        self.validation_ids = list(self.x_val['Article'])

        if self.verbose > 0:

            print('X_train_validation shape: {}'.format(self.x_train_validation.shape))
            print('y_train_validation shape: {}'.format(self.y_train_validation.shape))

            print('X_train shape: {}'.format(self.x_train.shape))
            print('y_train shape: {}'.format(self.y_train.shape))

            print('X_validation shape: {}'.format(self.x_val.shape))
            print('y_validation shape: {}'.format(self.y_val.shape))

            print('X_test shape: {}'.format(self.x_test.shape))

            print('\nTrain Dataset Label Ratios: ')
            for t in self.__calc_label_ratios(self.y_train):
                print("Label {}: {}%".format(t[0], t[1]))

            print('\nValidation Dataset Label Ratios: ')
            for t in self.__calc_label_ratios(self.y_val):
                print("Label {}: {}%".format(t[0], t[1]))

        self.__create_authors_label_props(train_val_enhanced)

        self.__create_authors_graph()

        return None


def dump_data_loader():
    """

    :return:
    """
    load_data_obj = DataLoader(verbose=0)
    load_data_obj.run_data_preparation()

    outfile = os.path.join(PROCESSED_DATA_DIR, 'DataLoader.pickle')
    with open(outfile, 'wb') as handle:
        pickle.dump(load_data_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('DataLoader object dumped at: {}'.format(outfile))


def restore_data_loader():
    """

    :return:
    """
    infile = os.path.join(PROCESSED_DATA_DIR, 'DataLoader.pickle')

    with open(infile, 'rb') as handle:
        b = pickle.load(handle)

    return b


if __name__ == "__main__":
    dump_data_loader()
    obj = restore_data_loader()
