from sklearn.preprocessing import LabelBinarizer

from rpcc.create_features import TextFeaturesExtractor, GraphFeaturesExtractor
from rpcc.load_data import DataLoader


if __name__ == "__main__":
    dl_obj = DataLoader()
    dl_obj.run_data_preparation()

    x_train = dl_obj.x_train['authors']
    y_train = dl_obj.y_train

    authors = x_train

    tfe_obj = TextFeaturesExtractor(authors)

    authors_graph = tfe_obj.create_authors_graph(authors)

    gfe_obj = GraphFeaturesExtractor(authors_graph)

    communities = gfe_obj.get_one_hot_communities

    print(communities.columns)

    lb = LabelBinarizer()

    # y_train_one_hot = lb.fit_transform(y_train)
    #
    # print(y_train_one_hot.shape)
    #
    # ab_emb_obj = FeedForward(emb_size=200,
    #                          voc_size=100,
    #                          max_sequence_length=1)
    #
    # ab_emb_obj.build_model()
    #
    # ab_emb_obj.fit(X=x_train,
    #                y=y_train_one_hot,
    #                epochs=100,
    #                val_size=0.2,
    #                bs=128,
    #                lr=0.01)
    #
    # ab_emb_obj.model.save('auth_communities.h5')
