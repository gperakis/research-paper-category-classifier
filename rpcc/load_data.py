import os
import pickle
from collections import Counter

import networkx as nx
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from rpcc import RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataLoader:

    def __init__(self,
                 train_file_name: str = 'train.csv',
                 test_file_name: str = 'test.csv',
                 graph_file_name: str = 'Cit-HepTh.txt',
                 info_file_name: str = "node_information.csv",
                 verbose: int = 0):
        """
        :type verbose: int
        :type info_file_name: str
        :type graph_file_name: str
        :type test_file_name: str
        :type train_file_name: str
        :param train_file_name: File name of the training data (features: article, journal)
        :param test_file_name: File name of the test data (features: article)
        :param graph_file_name: File name of the network structure
        :param info_file_name: File name of the data (features: article, title, year, authors, abstract)
        :param verbose: Level of verbosity
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

    def get_stratified_data(self,
                            X: pd.DataFrame,
                            y: pd.Series,
                            val_size: float = 0.2,
                            random_state: int = 0) -> dict:
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
    def __calc_label_ratios(x: iter) -> list:
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

    @property
    def create_citation_network(self) -> nx.DiGraph:
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

        return G

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
        # creating the abstract graphs for the train-validation abstracts
        train_val_enhanced['abstract_graph'] = train_val_enhanced['abstract'].apply(self.generate_graph_from_text)

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
        # creating the abstract graphs for the testing abstracts
        test_x['abstract_graph'] = test_x['abstract'].apply(self.generate_graph_from_text)

        test_x.rename(columns={'id': 'Article'}, inplace=True)

        self.test_ids = test_ids
        self.x_test = test_x

    def run_data_preparation(self, val_size=0.2):
        """
        Creates all the needed datasets and networks for the given inputs.

        :param val_size: float. percentage of the validation data set
        :return:
        """

        # Load data about each article in a dataframe and set it to the constuctor
        self.__load_article_metadata()

        # Create a directed graph and store it in the constructor
        self.create_citation_network()

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
