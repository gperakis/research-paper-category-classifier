from itertools import combinations

import networkx as nx
from more_itertools import windowed, flatten
from sklearn.feature_extraction.text import CountVectorizer

from rpcc.load_data import restore_data_loader, DataLoader
from pprint import pprint


def create_graph(text, len_window=4, directed=True, stop_word='english'):
    """

    :param text:
    :param len_window:
    :param directed:
    :return:
    """
    assert directed in [True, False]

    tokenizer = CountVectorizer(stop_words=stop_word).build_analyzer()

    tokens = tokenizer(text)

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # All the magic is here!  Creates the weights indices.
    weight_index = list(flatten([list(range(1, i + 1)) for i in range(len_window - 1, 0, -1)]))
    for window in windowed(tokens, len_window):
        for num, comb in enumerate(combinations(window, 2)):
            weight = 1 / weight_index[num]

            # if there is already an edge between the two authors, add more weight.
            if G.has_edge(comb[0], comb[1]):
                G[comb[0]][comb[1]]['weight'] += weight
            else:
                G.add_edge(comb[0], comb[1], weight=weight)

    G.remove_edges_from(G.selfloop_edges())

    return G


def extract_k_core(graph):
    """

    :param graph:
    :return:
    """
    core = nx.k_core(g)
    return core.nodes()


if __name__ == "__main__":
    dl_obj = restore_data_loader()
    x_train_val_abstracts = dl_obj.x_train_validation['abstract']
    g = create_graph(x_train_val_abstracts[1], len_window=3, directed=True, stop_word='english')
    x = extract_k_core(g)
    pprint(x)
    print(len(x))

    g = create_graph(x_train_val_abstracts[1], len_window=3, directed=False, stop_word='english')
    x = extract_k_core(g)
    pprint(x)
    print(len(x))
