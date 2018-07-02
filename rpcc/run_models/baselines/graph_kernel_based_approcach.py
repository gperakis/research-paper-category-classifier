import networkx as nx
from grakel import GraphKernel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from rpcc.create_features import TextFeaturesExtractor
from rpcc.load_data import DataLoader

sp_kernel = GraphKernel(kernel={"name": "shortest_path",
                                'with_labels': False},
                        normalize=True)

dl_obj = DataLoader()
dl_obj.run_data_preparation()

# creating a label binarizer instance in order to convert the classes to one hot vectors
lb = LabelEncoder()

# extracting the train targets
y_train = dl_obj.y_train

# converting the train targets to one hot
y_train_one_hot = lb.fit_transform(y_train)

# extracting the train targets
y_val = dl_obj.y_val

# converting the train targets to one hot
y_val_one_hot = lb.transform(y_val)

x_train_titles = dl_obj.x_train['title']
x_val_titles = dl_obj.x_val['title']
x_test_titles = dl_obj.x_test['title']

tfe_obj = TextFeaturesExtractor(None)

REMOVE_STOP_WORDS = True
DIRECTED = False

x_train_title_graphs = list()
for text in dl_obj.x_train['title']:
    graph = tfe_obj.generate_graph_from_text(text=text,
                                             remove_stopwords=REMOVE_STOP_WORDS,
                                             directed=DIRECTED)

    inp = nx.to_dict_of_lists(graph)
    x_train_title_graphs.append([inp])

x_val_title_graphs = list()
for text in dl_obj.x_val['title']:
    graph = tfe_obj.generate_graph_from_text(text=text,
                                             remove_stopwords=REMOVE_STOP_WORDS,
                                             directed=DIRECTED)

    inp = nx.to_dict_of_lists(graph)
    x_val_title_graphs.append([inp])

x_test_title_graphs = list()
for text in dl_obj.x_test['title']:
    graph = tfe_obj.generate_graph_from_text(text=text,
                                             remove_stopwords=REMOVE_STOP_WORDS,
                                             directed=DIRECTED)

    inp = nx.to_dict_of_lists(graph)
    x_test_title_graphs.append([inp])

print(len(x_train_title_graphs))
print(len(x_val_title_graphs))

K_train = sp_kernel.fit_transform(x_train_title_graphs)
K_val = sp_kernel.transform(x_val_title_graphs)

# clf = SVC(kernel='precomputed')
clf = LogisticRegression()

clf.fit(K_train, y_train_one_hot)

y_pred = clf.predict(K_val)

from sklearn.metrics import accuracy_score

print("%2.2f %%" % (round(accuracy_score(y_val_one_hot, y_pred) * 100)))
