import os

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from dsce import setup_logger, PROCESSED_DATA_DIR
from dsce.load_data import restore_data_loader, DataLoader

logger = setup_logger(__name__)


class CalculateAvgNeighbourDegree(BaseEstimator, TransformerMixin):

    def __init__(self, graph):
        """

        :param graph:
        :param ids:
        """
        self.graph = graph

        self.avg_neig_deg = nx.average_neighbor_degree(self.graph)

    def transform(self, X=None, y=None):
        """

        :param X: Network X node IDs in a list
        :param y:
        :return:
        """

        if X is None:
            df = pd.DataFrame.from_dict(self.avg_neig_deg, orient='index')
            df.columns = ['avg_neigh_deg']
            return df

        series = pd.Series(X, name='node_ids')
        return series.apply(lambda x: self.avg_neig_deg[x])

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class CalculateOutDegree(BaseEstimator, TransformerMixin):

    def __init__(self, graph):
        """

        :param graph:
        """
        self.graph = graph

        self.out_degree = {tupl[0]: tupl[1] for tupl in self.graph.out_degree()}

    def transform(self, X=None, y=None):
        """

        :param X: Network X node IDs in a list
        :param y:
        :return:
        """

        if X is None:
            df = pd.DataFrame.from_dict(self.out_degree, orient='index')
            df.columns = ['out_degree']
            return df

        series = pd.Series(X, name='node_ids')
        return series.apply(lambda x: self.out_degree.get(x))

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class CalculateInDegree(BaseEstimator, TransformerMixin):

    def __init__(self, graph):
        """

        :param graph:
        """
        self.graph = graph

        self.in_degree = {tupl[0]: tupl[1] for tupl in self.graph.in_degree()}

    def transform(self, X=None, y=None):
        """

        :param X: Network X node IDs in a list
        :param y:
        :return:
        """
        if X is None:
            df = pd.DataFrame.from_dict(self.in_degree, orient='index')
            df.columns = ['in_degree']

            return df

        series = pd.Series(X, name='node_ids')
        return series.apply(lambda x: self.in_degree.get(x))

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class CalculateOutDegreeCentrality(BaseEstimator, TransformerMixin):

    def __init__(self, graph):
        """

        :param graph:
        """
        self.graph = graph

        self.out_deg_centrality = nx.out_degree_centrality(self.graph)

    def transform(self, X=None, y=None):
        """

        :param X: Network X node IDs in a list
        :param y:
        :return:
        """

        if X is None:
            df = pd.DataFrame.from_dict(self.out_deg_centrality, orient='index')
            df.columns = ['out_degree_centr']
            return df

        series = pd.Series(X, name='node_ids')

        return series.apply(lambda x: self.out_deg_centrality[x])

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class CalculateInDegreeCentrality(BaseEstimator, TransformerMixin):

    def __init__(self, graph):
        """

        :param graph:
        """
        self.graph = graph

        self.in_deg_centrality = nx.in_degree_centrality(self.graph)

    def transform(self, X=None, y=None):
        """

        :param X: Network X node IDs in a list
        :param y:
        :return:
        """

        if X is None:
            df = pd.DataFrame.from_dict(self.in_deg_centrality, orient='index')
            df.columns = ['in_degree_centr']
            return df

        series = pd.Series(X, name='node_ids')

        return series.apply(lambda x: self.in_deg_centrality[x])

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class CalculateBetweenessCentrality(BaseEstimator, TransformerMixin):

    def __init__(self, graph):
        """

        :param graph:
        """
        self.graph = graph

        self.betweeness_centrality = nx.betweenness_centrality(self.graph)

    def transform(self, X=None, y=None):
        """

        :param X: Network X node IDs in a list
        :param y:
        :return:
        """

        if X is None:
            df = pd.DataFrame.from_dict(self.betweeness_centrality, orient='index')
            df.columns = ['betweeness_centr']

            return df

        series = pd.Series(X, name='node_ids')

        return series.apply(lambda x: self.betweeness_centrality[x])

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class CalculateClosenessCentrality(BaseEstimator, TransformerMixin):

    def __init__(self, graph):
        """

        :param graph:
        """
        self.graph = graph

        self.closeness_centrality = nx.closeness_centrality(self.graph)

    def transform(self, X=None, y=None):
        """

        :param X: Network X node IDs in a list
        :param y:
        :return:
        """

        if X is None:
            df = pd.DataFrame.from_dict(self.closeness_centrality, orient='index')
            df.columns = ['closeness_centr']

            return df

        series = pd.Series(X, name='node_ids')

        return series.apply(lambda x: self.closeness_centrality[x])

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class CalculatePageRank(BaseEstimator, TransformerMixin):

    def __init__(self, graph):
        """

        :param graph:
        """
        self.graph = graph

        self.page_rank = pr = nx.pagerank(self.graph, alpha=0.9)

    def transform(self, X=None, y=None):
        """

        :param X: Network X node IDs in a list
        :param y:
        :return:
        """

        if X is None:
            df = pd.DataFrame.from_dict(self.page_rank, orient='index')
            df.columns = ['page_rank']

            return df

        series = pd.Series(X, name='node_ids')

        return series.apply(lambda x: self.page_rank[x])

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class CalculateHubsAndAuthorities(BaseEstimator, TransformerMixin):
    """Works for directed graphs only"""

    def __init__(self, graph):
        """

        :param graph:
        """
        self.graph = graph

        self.hubs, self.authorities = nx.hits(self.graph)

    def transform(self, X, y=None):
        """

        :param X: Network X node IDs in a list
        :param y:
        :return:
        """

        if X is None:
            all_nodes = self.graph.nodes()
            df = pd.DataFrame(index=all_nodes)

            df['hubs'] = pd.Series(df.index).apply(lambda x: self.hubs[x])
            df['authorities'] = pd.Series(df.index).apply(lambda x: self.authorities[x])

            return df

        df = pd.DataFrame(X, columns=['node_ids'])
        df['hubs'] = df['node_ids'].apply(lambda x: self.hubs[x])
        df['authorities'] = df['node_ids'].apply(lambda x: self.authorities[x])

        return df[['hubs', 'authorities']]

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class CalculateNumberOfTriangles(BaseEstimator, TransformerMixin):
    def __init__(self, graph):
        """

        :param graph:
        """
        self.graph = graph

        self.n_triangles = nx.triangles(self.graph)

    def transform(self, X=None, y=None):
        """

        :param X: Network X node IDs in a list
        :param y:
        :return:
        """
        if X is None:
            df = pd.DataFrame.from_dict(self.n_triangles, orient='index')

            df.columns = ['n_triangles']

            return df

        series = pd.Series(X, name='node_ids')

        return series.apply(lambda x: self.n_triangles[x])

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


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


def create_paper_author_features(papers_df, authors_probs, targets, authors_graph):
    """

    :param papers_df:
    :param authors_probs:
    :param targets:
    :param authors_graph:
    :return:
    """
    docs = papers_df[['Article', 'authors']].to_dict('records')

    results = list()

    for doc in tqdm(docs):
        authors = DataLoader.clean_up_authors(doc['authors'])
        # only keeping those authors that have length over 2 characters.
        co_authors = [author for author in authors if len(author) > 2]

        paper_features = {'Article': doc['Article']}

        if co_authors:
            paper_features = get_paper_authors_metadata(co_authors,
                                                        authors_probs,
                                                        targets,
                                                        authors_graph)
        results.append(paper_features)

    return pd.DataFrame(results)


def create_cites_graph_features(load=False, save=True):
    """

    :param load:
    :param save:
    :return:
    """
    outfile = os.path.join(PROCESSED_DATA_DIR, 'cites_graph_features_df.csv')

    if load:
        df = pd.read_csv(outfile)
        return df

    dl_obj = restore_data_loader()
    cites_graph = dl_obj.cites_graph

    avg_neigh_deg = CalculateAvgNeighbourDegree(graph=cites_graph).fit_transform(X=None)
    out_deg_dict = CalculateOutDegree(graph=cites_graph).fit_transform(X=None)
    in_deg_dict = CalculateInDegree(graph=cites_graph).fit_transform(X=None)
    out_deg_centrality = CalculateOutDegreeCentrality(graph=cites_graph).fit_transform(X=None)
    in_deg_centrality = CalculateInDegreeCentrality(graph=cites_graph).fit_transform(X=None)
    betweeness_centrality = CalculateBetweenessCentrality(graph=cites_graph).fit_transform(X=None)
    closeness_centrality = CalculateClosenessCentrality(graph=cites_graph).fit_transform(X=None)
    page_rank = CalculatePageRank(graph=cites_graph).fit_transform(X=None)
    hubs_authorities = CalculateHubsAndAuthorities(graph=cites_graph).fit_transform(X=None)
    triangles = CalculateNumberOfTriangles(graph=cites_graph).fit_transform(X=None)

    features = [
        avg_neigh_deg,
        out_deg_dict,
        in_deg_dict,
        out_deg_centrality,
        in_deg_centrality,
        betweeness_centrality,
        closeness_centrality,
        page_rank,
        hubs_authorities,
        triangles
    ]

    df = pd.DataFrame(index=cites_graph.nodes())

    for feature_df in features:
        df = df.merge(feature_df, left_index=True, right_index=True)

    df.reset_index(inplace=True)

    if save:
        df.to_csv(outfile, encoding='utf-8', index=True)

    return df


if __name__ == "__main__":
    cites_df = create_cites_graph_features(load=False, save=True)

    print(cites_df)

    # print(obj2.out_degree)

    # authors = ['P.H. Damgaard', 'Yuri A. Kubyshin', 'S.P. Khastgir']

    # x = create_paper_author_features(dl_obj.x_val,
    #                                  dl_obj.authors_label_props,
    #                                  dl_obj.targets,
    #                                  authors_graph)
    #
    # pprint(x)
