import operator
from pickle import dump

from rpcc import MODELS_DIR
from rpcc.create_features import FeatureExtractor
from rpcc.load_data import DataLoader
from rpcc.models import FeedForward
from sklearn.model_selection import KFold

"""
This script is responsible for running the hyperparameter tuning.

"""


def select_features(X, setting):
    """
    It selects specific features according to the given setting

    :param X: Pandas Dataframe, the training data
    :param setting: list of strings, with the columns to be selected
    :return: Pandas Dataframe, the training data with columns according to the setting
    """
    return X


def cross_validation(data):
    """
    It splits the data into folds and run the model training for each fold.

    :param data:
    :return:
    """
    n_splits = 3

    kf = KFold(n_splits)

    model = None
    performance = 0

    for train, dev in kf.split(data):
        model = FeedForward(100, 100, 100)
        model.fit(train, train)

        # TODO predict should return the evaluation of the model and
        # TODO if dump=true should write the predictions to a file
        performance += model.predict(dev)

    performance /= n_splits

    return model.features, performance


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
    dataset = DataLoader().data['train']

    # transform the data in order to get the desired features
    transformed_data = FeatureExtractor(dataset)

    # set the different feature combinations you want to test
    settings = {'run_a': list(),
                'run_b': list()}

    hyper_parameter_results = dict()

    for setting in settings:
        hyper_parameter_results[setting] = dict()

        subset_of_data = select_features(transformed_data, setting)

        # perform cross-validation
        feature_set, results = cross_validation(subset_of_data, subset_of_data)

        hyper_parameter_results[setting]['feature_set'] = feature_set
        hyper_parameter_results[setting]['results'] = results

    final_settings = select_best_model_features(hyper_parameter_results)

    # run the best model to all the data
    subset_of_data = select_features(transformed_data, final_settings)

    final_model = FeedForward(100, 100, 100)
    final_model.fit(subset_of_data, subset_of_data)

    dump_model(final_model)
