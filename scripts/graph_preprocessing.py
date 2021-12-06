from networkx.classes.function import nodes
import numpy as np
import argparse
import psycopg2
import pickle
import zlib
import os
import warnings
import json
from psycopg2 import sql
from sqlalchemy import create_engine
import networkx as nx
import trackintel as ti
from pyproj import Transformer, CRS


def read_graphs_from_postgresql(
    graph_table_name,
    psycopg_con,
    graph_schema_name="public",
    file_name="graph_data",
    decompress=True,
):
    """
    reads `graph_data` from postgresql database. Reads a single row named
     `graph_data` from `schema_name`.`table_name`

    Parameters
    ----------
    graph_data: Dictionary of activity graphs to be stored
    graph_table_name: str
        Name of graph in database. Corresponds to the row name.
    graph_schema_name: str
        schema name for table
    file_name: str
        Name of row to store the graph (identifier)
    decompress

    Returns
    -------

    """
    # retrieve string
    cur = psycopg_con.cursor()
    cur.execute(
        sql.SQL("select data from {}.{} where name = %s").format(
            sql.Identifier(graph_schema_name), sql.Identifier(graph_table_name)
        ),
        (file_name,),
    )
    pickle_string2 = cur.fetchall()[0][0].tobytes()

    cur.close()
    if decompress:
        AG_dict2 = pickle.loads(zlib.decompress(pickle_string2))
    else:
        AG_dict2 = pickle.loads(pickle_string2)

    return AG_dict2


def get_con():
    DBLOGIN_FILE = os.path.join("dblogin.json")
    with open(DBLOGIN_FILE) as json_file:
        LOGIN_DATA = json.load(json_file)

    con = psycopg2.connect(
        dbname=LOGIN_DATA["database"],
        user=LOGIN_DATA["user"],
        password=LOGIN_DATA["password"],
        host=LOGIN_DATA["host"],
        port=LOGIN_DATA["port"],
    )
    return con


def _load_graphs(study, node_importance=0, remove_loops=True):
    con = get_con()
    # Maybe need to use the function _get_db_params again (from graph_features)
    table_name, file_name = ("full_graph", "graph_data")
    # table_name, study_for_db, file_name = self._get_db_params(study)
    graph_dict = read_graphs_from_postgresql(
        graph_table_name=table_name,
        psycopg_con=con,
        graph_schema_name=study,
        file_name=file_name,
        decompress=True,
    )
    users = []
    nx_graphs = []
    for user_id, ag in graph_dict.items():
        if node_importance == 0:
            ag_sub = ag.G
        else:
            important_nodes = ag.get_k_importance_nodes(node_importance)
            ag_sub = nx.DiGraph(ag.G.subgraph(important_nodes))

        # delete edges with transition weight 0:
        edges_to_delete = [
            (a, b)
            for a, b, attrs in ag_sub.edges(data=True)
            if attrs["weight"] < 1
        ]
        if len(edges_to_delete) > 0:
            ag_sub.remove_edges_from(edges_to_delete)
        if ag_sub.number_of_edges() == 0:
            print("zero edges for user", user_id, " --> skip!")
            continue

        # get only the largest connected component:
        cc = sorted(
            nx.connected_components(ag_sub.to_undirected()),
            key=len,
            reverse=True,
        )
        graph_cleaned = ag_sub.subgraph(cc[0])

        # filter self loops (Probably required for GCN)
        if remove_loops:
            graph_cleaned = nx.DiGraph(graph_cleaned)
            graph_cleaned.remove_edges_from(nx.selfloop_edges(graph_cleaned))

        users.append(user_id)
        nx_graphs.append(graph_cleaned)
    return nx_graphs, users


def graph_preprocessing(graph, number_of_nodes=0):
    """
    Preprocess the graphs by projecting the coordinates and return only the
    coordinates and adjacency matrix
    """
    # filter for the nodes with largest degrees (not done if number_of_nodes=0)
    if number_of_nodes > 0:
        sorted_dict = np.array(
            [
                [k, v]
                for k, v in sorted(
                    dict(graph.degree()).items(),
                    key=lambda item: item[1],
                )
            ]
        )
        use_nodes = sorted_dict[-number_of_nodes:, 0]
        graph = graph.subgraph(use_nodes)

    # project coordinates and filter out nodes that are too far away
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:21781", always_xy=True
    )
    # get bounds
    out_crs = CRS.from_epsg(21781)
    x_min, y_min, x_max, y_max = out_crs.area_of_use.bounds
    # iterate over nodes
    nodes_inside_bounds = []
    node_loc_arr = []
    for node in graph.nodes():
        center = graph.nodes[node]["center"]
        if (x_min < center.x < x_max) and (y_min < center.y < y_max):
            transformed_center = transformer.transform(center.x, center.y)
            node_loc_arr.append(list(transformed_center))
            nodes_inside_bounds.append(node)
    # restrict to the nodes in switzerland
    graph = graph.subgraph(nodes_inside_bounds)

    # No nodes left
    if len(nodes_inside_bounds) == 0:
        warnings.warn("No nodes left in switzerland")
        return np.zeros((0, 0)), np.zeros((0, 0))

    node_loc_arr = np.array(node_loc_arr)

    # compute the weighted indegree for each node
    weighted_adjacency = nx.linalg.graphmatrix.adjacency_matrix(
        graph, weight="weight"
    )

    return node_loc_arr, weighted_adjacency


if __name__ == "__main__":

    number_of_nodes = 0  # get all nodes (except for isolates)

    users, adjacencies, node_feats = [], [], []
    for study in ["geolife"]:  # ["gc2", "gc1", "yumuv_graph_rep"]:
        print(f"------ Process {study} ---------")
        graph_list, users_study = _load_graphs(study, node_importance=0)

        # extract features and adjacency from individual graphs
        for graph, u in zip(graph_list, users_study):
            graph_node_feats, graph_adjacency = graph_preprocessing(
                graph, number_of_nodes=number_of_nodes
            )
            if len(graph_node_feats) == 0:
                print("No nodes left for user", u)
                continue
            node_feats.append(graph_node_feats)
            adjacencies.append(graph_adjacency)
            users.append(study + "_" + str(u))

    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "test_data.pkl"), "wb") as outfile:
        pickle.dump((users, adjacencies, node_feats), outfile)