import numpy as np
import argparse
import psycopg2
import pickle
import zlib
import os
import json
from psycopg2 import sql
from sqlalchemy import create_engine
import networkx as nx
import trackintel as ti


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


if __name__ == "__main__":

    number_of_nodes = 50

    users, adjacencies, node_feats = [], [], []
    for study in ["geolife"]:  # ["gc2", "gc1", "yumuv_graph_rep"]:
        print(f"------ Process {study} ---------")
        graph_list, users_study = _load_graphs(study, node_importance=0)
        users.extend([study + "_" + str(u) for u in users_study])

        # extract features and adjacency from individual graphs
        for graph in graph_list:
            # filter for the nodes with largest degrees
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

            # append unweighted adjacency matrix
            adjacencies.append(
                nx.linalg.graphmatrix.adjacency_matrix(graph, weight=None)
            )
            # compute the weightde indegree for each node
            weighted_adjacency = nx.linalg.graphmatrix.adjacency_matrix(
                graph, weight="weight"
            )
            label = np.array(np.sum(weighted_adjacency, axis=0))[0]

            # find home node
            all_degrees = np.array(graph.out_degree())
            home_node = all_degrees[np.argmax(all_degrees[:, 1]), 0]
            home_center = graph.nodes[home_node]["center"]

            # compute coordinate features for other nodes
            node_loc_arr = []
            for node_ind, node in enumerate(graph.nodes()):
                center = graph.nodes[node]["center"]
                dist = ti.geogr.point_distances.haversine_dist(
                    center.x, center.y, home_center.x, home_center.y
                )[0]
                # save distance and relative displacement vector
                node_loc_arr.append(
                    [
                        dist,
                        center.x - home_center.x,
                        center.y - home_center.y,
                        label[node_ind],
                    ]
                )
            node_loc_arr = np.array(node_loc_arr)
            # print("node features size", node_loc_arr.shape)
            # print(np.min(label))

            node_feats.append(node_loc_arr)

    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "test_data.pkl"), "wb") as outfile:
        pickle.dump((users, adjacencies, node_feats), outfile)
