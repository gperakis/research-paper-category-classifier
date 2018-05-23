import networkx as nx
from rpcc.load_data import restore_data_loader, DataLoader
import itertools
from more_itertools import windowed


def create_graph(text):
    """

    :param text:
    :return:
    """
    # instantiating a simple Graph.
    G = nx.DiGraph()
    tokens = text.split()

    # creates a sliding window of text tokens
    for window in windowed(tokens, 3):
        # creates combinations of tokens of size 2
        for nodes_tpl in itertools.combinations(window, 2):

            # calculates the weight that we want to give between the nodes.
            try:
                weight = 1 / (window.index(nodes_tpl[1]) - window.index(nodes_tpl[0]))

            except ZeroDivisionError:
                indices = [i[0] for i in enumerate(window) if i[1] == nodes_tpl[0]]

                if len(indices) == 2:
                    weight = 1 / (indices[1] - indices[0])

                else:
                    # we set it to -1 in order to not add the edge using an if statement
                    weight = -1

            if weight >= 0:
                # if there is already an weighted edge between the two authors, add more weight.
                if G.has_edge(nodes_tpl[0], nodes_tpl[1]):

                    G[nodes_tpl[0]][nodes_tpl[1]]['weight'] += weight

                else:
                    # create the connection between the nodes
                    G.add_edge(nodes_tpl[0], nodes_tpl[1], weight=weight)

            else:
                print(nodes_tpl[0], nodes_tpl[1])

    return G


if __name__ == "__main__":

    dl_obj = restore_data_loader()
    abstracts = dl_obj.x_train_validation['abstract']

    for abs in abstracts:
        create_graph(abs)
