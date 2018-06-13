

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
