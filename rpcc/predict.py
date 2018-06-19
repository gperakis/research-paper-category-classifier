from pickle import load

from rpcc import MODELS_DIR
from rpcc.create_features import FeatureExtractor
from rpcc.hyperparameter_tuning import select_features
from rpcc.load_data import DataLoader

"""
This script is responsible for the model predictions on the test dataset.

After the test data is loaded, they should be transformed into a Pandas
DataFrame with the same features as our models is build upon.

Then the script loads the final model that will be used for the prediction.
This Model object will have stored the features that was trained and
the trained network.

Based on that features, the final_test_data dataset is created and is feeded to
model's predictor.

"""


if __name__ == '__main__':

    # import all the data (train & dev) in a pandas DataFrame
    data_obj = DataLoader()
    test_data = data_obj.data['test']

    # transform the data in order to get the desired features
    transformed_test_data = FeatureExtractor(test_data)

    # load the best model
    file = open(MODELS_DIR + "final_model.pickle", 'rb')
    model = load(file)

    # transform data to have the required features
    final_test_data = select_features(transformed_test_data, model.settings)

    # perform predictions
    model.predict(final_test_data)

