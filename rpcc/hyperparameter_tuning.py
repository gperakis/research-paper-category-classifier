import operator
from pickle import dump

from rpcc import MODELS_DIR
# from rpcc.create_features import FeatureExtractor
from rpcc.load_data import DataLoader
from rpcc.models import FeedForward
from sklearn.model_selection import KFold
from rpcc.case_class import x, test_x, y
import pandas as pd
from pprint import pprint
import numpy as np

"""
This script is responsible for running the hyperparameter tuning.

"""


def select_features(data, settings):
    """
    It selects specific features according to the given setting

    :param data: Pandas Dataframe, the training data
    :param setting: list of strings, with the columns to be selected
    :return: Pandas Dataframe, the training data with columns according to the setting
    """
    columns_to_select = list()
    for column in list(data.columns.values):
        for set in settings:
            if column.startswith(set):
                columns_to_select.append(column)

    return data[list(columns_to_select)]


def cross_validation(x_data, y_data):
    """
    It splits the data into folds and run the model training for each fold.

    :param data:
    :return:
    """
    n_splits = 3

    kf = KFold(n_splits)

    performance = 0

    for train, dev in kf.split(x_data):
        x_train = x_data.iloc[train].values
        y_train = y_data.iloc[train].values

        x_dev = x_data.iloc[dev].values
        y_dev = y_data.iloc[dev].values

        model = FeedForward(emb_size=100,
                            voc_size=100,
                            max_sequence_length=len(list(x_data.columns.values)))
        model.fit(x_train,
                  y_train)

        score = model.predict(x_dev, y_dev)
        performance += score['scores']['acc']

    performance /= n_splits

    return performance


def select_best_model_features(models_run):
    """
    It sorts the given models based on their performance and returns the best Model object
    with its setting of features.

    :param models_run: python dictionary, with Model objects
    :return: Model object
    """
    best_features = max(models_run.items(), key=operator.itemgetter(1))[0]

    return best_features


def dump_model(model):
    """
    It saves a given model to a pickle file

    :param model: a Model object
    """
    with open(MODELS_DIR + 'final_model.pickle', "wb") as output_file:
        dump(model, output_file)


if __name__ == '__main__':

    # import all the data (train & dev) in a pandas DataFrame
    # dataset = DataLoader().data['train']

    # transform the data in order to get the desired features
    # transformed_data = FeatureExtractor(dataset)

    transformed_data = x

    # set the different feature combinations you want to test
    settings = [('graph', 'author'),
                ('abstract', 'title')]

    hyper_parameter_results = dict()

    for setting in settings:
        print('Running for features: {}'.format(setting))

        subset_of_data = select_features(transformed_data, setting)

        # perform cross-validation
        results = cross_validation(subset_of_data, y)

        print('-' * 50 + 'End of setting pass. {} {}'.format(setting, results))

        hyper_parameter_results[setting] = results

    pprint(hyper_parameter_results)

    final_settings = select_best_model_features(hyper_parameter_results)

    print(transformed_data)
    print(final_settings)

    # run the best model to all the data
    subset_of_data = select_features(transformed_data, final_settings)

    y_fit = y.values
    X_fit = subset_of_data.values

    X_test = select_features(test_x, final_settings).values
    y_test = np.array(list(map(lambda x: 0 if x < 0.5 else 1, np.random.rand(len(X_test), 1))))

    print(X_fit)
    print(y_fit)
    print(X_test)
    print(y_test)

    final_model = FeedForward(100, 100, len(list(subset_of_data.columns.values)))
    final_model.fit(X_fit, y_fit)

    # dump_model(final_model)

    scores = final_model.predict(X_test, y_test, dump=True)
