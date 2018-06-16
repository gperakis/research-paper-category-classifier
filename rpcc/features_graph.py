import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite
from tqdm import tqdm

from rpcc import RAW_DATA_DIR
from rpcc import setup_logger, PROCESSED_DATA_DIR
from rpcc.load_data import restore_data_loader, DataLoader

logger = setup_logger(__name__)


def normalize_dict_values(d):
    """

    :param d:
    :return:
    """

    sum_values = sum(d.values())

    if sum_values > 0:
        factor = 1.0 / sum_values

        d = {k: v * factor for k, v in d.items()}

    return d


def get_cliques_metadata(G, node, targets, props=None):
    """

    :param G:
    :param node:
    :param props:
    :return:
    """

    out = {'node_cliques_size_avg': 0,
           'node_cliques_size_std': 0,
           'node_cliques_size_max': 0,
           'node_cliques_number': 0,
           'node_cliques_props_avg': {}}

    cliques = nx.cliques_containing_node(G, nodes=node)

    if cliques:
        clique_sizes = [len(c) for c in cliques]

        out['node_cliques_size_avg'] = np.mean(clique_sizes)
        out['node_cliques_size_std'] = np.std(clique_sizes)
        out['node_cliques_size_max'] = max(clique_sizes)
        out['node_cliques_number'] = len(cliques)

        if props:

            all_clique_props = list()

            for clique in cliques:
                clique_props = list()
                for n in clique:
                    clique_props.append(props.get(n, dict.fromkeys(targets, 0.0)))

                clique_avg_label_props = pd.DataFrame(clique_props).sum(axis=0).to_dict()
                clique_avg_label_props = normalize_dict_values(clique_avg_label_props)

                all_clique_props.append(clique_avg_label_props)

            # taking thee average for all cliques.
            avg_cliques_label_props = (pd.DataFrame(all_clique_props).sum(axis=0) / len(all_clique_props)).to_dict()
            # normalizing the values of the dictionary
            avg_cliques_label_props = normalize_dict_values(avg_cliques_label_props)

            out['node_cliques_props_avg'] = avg_cliques_label_props

    return out


def get_paper_authors_metadata(authors, authors_probs, targets, authors_graph):
    """

    :param authors:
    :param authors_probs:
    :param targets:
    :param authors_graph:
    :return:
    """

    out = dict()

    probs_results = list()
    clique_max_sizes = list()
    node_cliques_number_results = list()
    average_clique_sizes_results = list()
    std_clique_sizes_results = list()
    for author in authors:
        x = get_cliques_metadata(authors_graph,
                                 node=author,
                                 targets=targets,
                                 props=authors_probs)

        probs_results.append(x['node_cliques_props_avg'])
        node_cliques_number_results.append(x['node_cliques_number'])
        average_clique_sizes_results.append(x['node_cliques_size_avg'])

        clique_max_sizes.append(x['node_cliques_size_max'])
        std_clique_sizes_results.append(x['node_cliques_size_std'])

    avg_props = (pd.DataFrame(probs_results).sum(axis=0) / len(probs_results)).to_dict()
    avg_props = normalize_dict_values(avg_props)

    total_max = max(clique_max_sizes)
    avg_max = np.mean(clique_max_sizes)
    std_max = np.std(clique_max_sizes)

    paper_avg_number_of_cliques = np.mean(node_cliques_number_results)
    paper_std_number_of_cliques = np.std(node_cliques_number_results)
    paper_max_number_of_cliques = max(node_cliques_number_results)

    paper_mean_average_clique_size = np.mean(average_clique_sizes_results)
    paper_std_average_clique_size = np.std(average_clique_sizes_results)

    paper_mean_std_clique_size = np.mean(std_clique_sizes_results)
    paper_std_std_clique_size = np.std(std_clique_sizes_results)

    out.update(avg_props)

    out['paper_max_clique_size'] = total_max
    out['paper_avg_max_clique_size'] = avg_max
    out['paper_std_max_clique_size'] = std_max

    out['paper_avg_number_of_cliques'] = paper_avg_number_of_cliques
    out['paper_std_number_of_cliques'] = paper_std_number_of_cliques
    out['paper_max_number_of_cliques'] = paper_max_number_of_cliques

    out['paper_mean_average_clique_size'] = paper_mean_average_clique_size
    out['paper_std_average_clique_size'] = paper_std_average_clique_size

    out['paper_mean_std_clique_size'] = paper_mean_std_clique_size
    out['paper_std_std_clique_size'] = paper_std_std_clique_size

    return out


def create_paper_author_features(load=False, save=True):
    """

    :param load:
    :param save:
    :return:
    """
    outfile = os.path.join(PROCESSED_DATA_DIR, 'authors_graph_features.csv')

    if load:
        df = pd.read_csv(outfile)
        return df

    dl_obj = restore_data_loader()

    authors_probs = dl_obj.authors_label_props
    targets = dl_obj.targets
    authors_graph = dl_obj.authors_graph
    papers_df = pd.concat([dl_obj.x_train_validation, dl_obj.x_test])[['Article', 'authors']]

    docs = papers_df[['Article', 'authors']].to_dict('records')

    results = list()

    counter = 0
    for doc in tqdm(docs):
        article_id = str(doc['Article'])
        authors = DataLoader.clean_up_authors(doc['authors'])
        # only keeping those authors that have length over 2 characters.
        co_authors = [author for author in authors if len(author) > 2]

        paper_features = {'Article': article_id}

        if co_authors:
            paper_features.update(get_paper_authors_metadata(co_authors,
                                                             authors_probs,
                                                             targets,
                                                             authors_graph))
        results.append(paper_features)

        if save:
            counter += 1

            if counter % 100 == 0:
                df = pd.DataFrame(results)
                df.to_csv(outfile, encoding='utf-8', index=False)
                print('Saved {} paper features'.format(counter))

    df = pd.DataFrame(results)
    if save:
        df.to_csv(outfile, encoding='utf-8', index=False)

    return df


def create_author_graph_features(load=False, save=True):
    """

    :param load:
    :param save:
    :return:
    """
    outfile = os.path.join(PROCESSED_DATA_DIR, 'pure_author_graph_features.csv')

    if load:
        df = pd.read_csv(outfile)
        return df

    dl_obj = restore_data_loader()
    cites_graph = dl_obj.authors_graph

    features_objects = [
        CalculateAvgNeighbourDegree(graph=cites_graph),
        CalculateUndirectedDegree(graph=cites_graph),
        CalculateUndirectedDegreeCentrality(graph=cites_graph),
        CalculateNumberOfTriangles(graph=cites_graph, to_undirected=False),
        CalculatePageRank(graph=cites_graph),
        CalculateHubsAndAuthorities(graph=cites_graph),
        CalculateBetweenessCentrality(graph=cites_graph),
        CalculateClosenessCentrality(graph=cites_graph),
    ]

    df = pd.DataFrame(index=cites_graph.nodes())

    for feat_obj in features_objects:
        feat_obj_df = feat_obj.fit_transform(X=None)
        df = df.merge(feat_obj_df, left_index=True, right_index=True)

        if save:
            df_out = df.reset_index()
            df_out.rename(columns={'index': 'author'}, inplace=True)
            df_out.to_csv(outfile, encoding='utf-8', index=True)
            print('Saved Features: {}'.format(df_out.columns.tolist()))

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'author'}, inplace=True)
    return df


def create_cites_graph_features(load=False, save=True):
    """

    :param load:
    :param save:
    :return:
    """
    outfile = os.path.join(PROCESSED_DATA_DIR, 'cites_graph_features.csv')

    if load:
        df = pd.read_csv(outfile)
        return df

    dl_obj = restore_data_loader()
    cites_graph = dl_obj.cites_graph

    features_objects = [
        CalculateAvgNeighbourDegree(graph=cites_graph),
        CalculateOutDegree(graph=cites_graph),
        CalculateInDegree(graph=cites_graph),
        CalculateOutDegreeCentrality(graph=cites_graph),
        CalculateInDegreeCentrality(graph=cites_graph),
        CalculateNumberOfTriangles(graph=cites_graph, to_undirected=True),
        CalculateBetweenessCentrality(graph=cites_graph),
        CalculateClosenessCentrality(graph=cites_graph),
        CalculatePageRank(graph=cites_graph),
        CalculateHubsAndAuthorities(graph=cites_graph),
    ]

    df = pd.DataFrame(index=cites_graph.nodes())

    for feat_obj in features_objects:
        feat_obj_df = feat_obj.fit_transform(X=None)
        df = df.merge(feat_obj_df, left_index=True, right_index=True)

        if save:
            df_out = df.reset_index()
            df_out.rename(columns={'index': 'Article'}, inplace=True)
            df_out.to_csv(outfile, encoding='utf-8', index=True)
            print('Saved Features: {}'.format(df_out.columns.tolist()))

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Article'}, inplace=True)
    df.fillna(0, inplace=True)

    return df


def create_combined_paper_authors_graph_features(load=False, save=True):
    """

    :return:
    """
    outfile = os.path.join(PROCESSED_DATA_DIR, 'total_paper_2_authors_features.csv')

    if load:
        df = pd.read_csv(outfile)
        return df

    dl_obj = restore_data_loader()
    papers_df = pd.concat([dl_obj.x_train_validation, dl_obj.x_test])[['Article', 'authors']]
    article_2_authors_dict = papers_df.set_index('Article').to_dict('index')

    author_graph_based_features = create_author_graph_features(load=True)
    author_graph_based_features.drop('Unnamed: 0', axis=1, inplace=True)
    author_graph_based_features.set_index('author', inplace=True)

    authors_metadata_dict = author_graph_based_features.to_dict('index')

    paper_author_graph_based_features = create_paper_author_features(load=True)
    paper_author_graph_based_features.fillna(0.0, inplace=True)

    records = paper_author_graph_based_features.to_dict('records')

    out = list()
    for doc in tqdm(records):
        doc['Article'] = str(int(doc['Article']))
        authors = article_2_authors_dict.get(doc['Article'], {}).get('authors', '')
        authors = DataLoader.clean_up_authors(authors)
        # only keeping those authors that have length over 2 characters.
        co_authors = [author for author in authors if len(author) > 2]

        temp_df = pd.DataFrame([authors_metadata_dict.get(author, {}) for author in co_authors])
        temp_df.dropna(inplace=True)

        paper_sum = temp_df.sum().add_prefix('paper_sum_').to_dict()
        paper_avg = temp_df.mean().add_prefix('paper_avg_')
        paper_std = temp_df.std().add_prefix('paper_std_')

        doc.update(paper_sum)
        doc.update(paper_avg)
        doc.update(paper_std)
        out.append(doc)

    df = pd.DataFrame(out)
    df.fillna(0, inplace=True)

    if save:
        df.to_csv(outfile, encoding='utf-8', index=False)
        print('Saved Features: {}'.format(df.columns.tolist()))
    return df


def extract_k_core(graph):
    """
    This function extracts the nodes of the k-core sub-graph of a given graph.
    :param graph:
    :return:
    """
    core = nx.k_core(graph)
    return core.nodes()


def create_authors2article_bipartite_graph(df):
    """

    :return:
    """

    B = nx.Graph()

    records = df.to_dict('records')

    author_tokenizer = DataLoader.clean_up_authors
    for rec in records:
        cleaned_authors = list(filter(lambda x: len(x) > 2,
                                      author_tokenizer(rec['authors'])))

        if cleaned_authors:
            weight = 1 / len(cleaned_authors)

            edge_list = [(author, rec['id']) for author in cleaned_authors]
            # B.add_nodes_from(cleaned_authors, bipartite=0)
            # B.add_nodes_from([rec['id']], bipartite=1)
            B.add_edges_from(edge_list, weight=weight)

    return B


def get_authors2titles(df):
    """

    :return:
    """

    records = df.to_dict('records')

    authors_resutls = list()
    author_tokenizer = DataLoader.clean_up_authors
    for rec in records:
        cleaned_authors = list(filter(lambda x: len(x) > 2,
                                      author_tokenizer(rec['authors'])))

        for auth in cleaned_authors:
            d = {'author': auth, 'title': rec['title']}
            authors_resutls.append(d)

    authors_df = pd.DataFrame(authors_resutls).dropna(subset=['title'])

    grouped = authors_df.groupby('author').agg({'title': lambda x: ' '.join(list(x))})

    return grouped



def plot_bipartite_graph(graph):
    """

    :param graph:
    :return:
    """
    x, y = bipartite.sets(graph)
    pos = dict()
    pos.update((n, (1, i)) for i, n in enumerate(x))  # put nodes from x at x=1
    pos.update((n, (2, i)) for i, n in enumerate(y))  # put nodes from y at x=2
    plt.figure(figsize=(40, 40))
    nx.draw(graph, pos=pos, with_labels=True)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    # plt.show()
    plt.savefig("Graph.png", format="PNG")


if __name__ == "__main__":
    df1 = pd.read_csv(os.path.join(RAW_DATA_DIR,
                                   'node_information.csv'), usecols=['id', 'authors', 'title'])
    df1 = df1.where((pd.notnull(df1)), None)

    # bipartite_graph = create_authors2article_bipartite_graph(df1)
    #
    # plot_bipartite_graph(bipartite_graph)

    get_authors2titles(df1)

    # cites_df = create_cites_graph_features(load=True, save=False)
    # print(cites_df)
    # x = create_combined_paper_authors_graph_features(load=True, save=False)
    # print(x.head())

    # dl_obj = restore_data_loader()
    # # citation_graph = dl_obj.citation_graph
    # print(dl_obj.article_metadata)

    # def get_community_labels(communities):
    #     """
    #
    #     :param communities: list of sets.
    #     :return: dict
    #     """
    #     out = dict()
    #     for num, doc in enumerate(communities):
    #         out.update(dict.fromkeys(doc, "community {}".format(num + 1)))
    #
    #     return out
    #
    #
    # communities_generator = community.girvan_newman(citation_graph)
    # first_level_communities = next(communities_generator)
    # second_level_communities = next(communities_generator)
    # third_level_communities = next(communities_generator)
    #
    # print(len(first_level_communities))
    # print(get_community_labels(first_level_communities))
    # print()
    #
    # print(len(second_level_communities))
    # print(get_community_labels(second_level_communities))
    # print()
    #
    # print(len(third_level_communities))
    # print(get_community_labels(third_level_communities))
    # print()
    # print(third_level_communities)
