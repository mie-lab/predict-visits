import os
import pickle
import numpy as np
import networkx as nx
import pandas as pd
from pyproj import Transformer, CRS
import copy
import json
import psycopg2
from tqdm import tqdm
from shapely import wkt, wkb

import trackintel as ti
from future_trackintel.activity_graph import activity_graph
from utils import (
    get_engine,
    get_staypoints,
    get_locations,
    get_triplegs,
    get_trips,
    filter_user_by_number_of_days,
)


CRS_WGS84 = "epsg:4326"

exclude_purpose_tist = [
    "Light Rail",
    "Subway",
    "Platform",
    "Trail",
    "Road",
    "Train",
    "Bus Line",
]


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


def download_data(study, engine, has_trips=True):
    """Download data of one study from database"""
    print("\t download staypoints")
    sp = get_staypoints(study=study, engine=engine, sp_name="staypoints_extent")

    print("\t download locations")
    sql = f"SELECT * FROM {study}.locations_extent"
    locs = ti.io.read_locations_postgis(sql, con=engine)
    locs["extent"] = locs["extent"].apply(lambda x: wkb.loads(bytes.fromhex(x)))

    gap_treshold = None
    trips = None
    # STUDIES WITH TRIPS
    if has_trips:
        print("\t download triplegs")
        tpls = get_triplegs(study=study, engine=engine)
        print("\t download trips")
        trips = get_trips(study=study, engine=engine)

        print("\t filter by tracking coverage")
        if study == "geolife":
            sp, user_id_ix = filter_user_by_number_of_days(
                sp=sp, tpls=tpls, coverage=0.7, min_nb_good_days=14
            )
        else:
            sp, user_id_ix = filter_user_by_number_of_days(
                sp=sp, tpls=tpls, coverage=0.7, min_nb_good_days=14
            )
        print("\t\t drop users with bad coverage")
        tpls = tpls[tpls.user_id.isin(user_id_ix)]
        trips = trips[trips.user_id.isin(user_id_ix)]
        locs = locs[locs.user_id.isin(user_id_ix)]
    # STUDIES WITHOUT TRIPS
    else:
        # exclude_purpose = ['Light Rail', 'Subway', 'Platform', 'Trail', 'Road', 'Train', 'Bus Line']
        sp = sp[~sp["purpose"].isin(exclude_purpose_tist)]
        gap_treshold = 12
        # a = pd.DataFrame(sp.groupby('purpose').size().sort_values()) TODO
    return (sp, locs, trips, gap_treshold)


def generate_graph(
    locs_user, sp_user, study, trips_user=None, gap_threshold=None
):
    """
    Given the locations and staypoints OF ONE USER, generate the graph
    """
    AG = activity_graph(
        sp_user,
        locs_user,
        trips=trips_user,
        gap_threshold=gap_threshold,
    )
    # Add purpose feature
    if study == "geolife":
        AG.add_node_features_from_staypoints(
            sp_user, agg_dict={"started_at": list, "finished_at": list}
        )
    else:
        AG.add_node_features_from_staypoints(
            sp_user,
            agg_dict={
                "started_at": list,
                "finished_at": list,
                "purpose": list,
            },
        )
    return AG


def delete_zero_edges(graph):
    edges_to_delete = [
        (a, b) for a, b, attrs in graph.edges(data=True) if attrs["weight"] < 1
    ]
    if len(edges_to_delete) > 0:
        graph.remove_edges_from(edges_to_delete)
    return graph


def get_largest_component(graph):
    """get only the largest connected component:"""
    cc = sorted(
        nx.connected_components(graph.to_undirected()),
        key=len,
        reverse=True,
    )
    graph_cleaned = graph.subgraph(cc[0])
    return graph_cleaned.copy()


def remove_loops(graph):
    graph.remove_edges_from(nx.selfloop_edges(nx.DiGraph(graph)))
    return graph


def keep_important_nodes(graph, number_of_nodes):
    """Reduce to the nodes with highest degree (in + out degree)"""
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
    return graph


def to_series(func):
    """Decorator to transform tuple into a series"""

    def add_series(center, home_center):
        normed_center = func(center.x, center.y, home_center)
        return pd.Series(normed_center, index=["x_normed", "y_normed"])

    return add_series


@to_series
def get_haversine_displacement(x, y, home_center):
    """Normalize (x, y) point by home center with haversine distance"""
    sign_x = 1 if x > home_center.x else -1
    displacement_x = ti.geogr.point_distances.haversine_dist(
        x, home_center.y, home_center.x, home_center.y
    )[0]
    sign_y = 1 if y > home_center.y else -1
    displacement_y = ti.geogr.point_distances.haversine_dist(
        home_center.x, y, home_center.x, home_center.y
    )[0]
    return displacement_x * sign_x, displacement_y * sign_y


def project_normalize_coordinates(node_feats, transformer=None, crs=None):
    """
    As input to the DL model, we want coordinates relative to home.
    To do so, we project the coordinates if possible or use the haversine
    distance.
    """
    # get home node:
    home_node = node_feats.iloc[
        (node_feats["in_degree"] + node_feats["out_degree"]).argmax()
    ]
    home_center = home_node["center"]

    @to_series
    def get_projected_displacement(x, y, home_center):
        if (x_min < x < x_max) and (y_min < y < y_max):
            proj_x, proj_y = transformer.transform(x, y)
            return (proj_x - home_center.x, proj_y - home_center.y)
        else:  # fall back to haversine
            return get_haversine_displacement(x, y, home_center)

    if transformer is not None:
        # get bounds
        x_min, y_min, x_max, y_max = crs.area_of_use.bounds
        normed_coords = node_feats["center"].apply(
            get_projected_displacement, args=[home_center]
        )
    else:
        normed_coords = node_feats["center"].apply(
            get_haversine_displacement, args=[home_center]
        )

    return pd.merge(
        node_feats, normed_coords, left_index=True, right_index=True
    )


def get_adj_and_attr(graph):
    list_of_nodes = list(graph.nodes())

    # get adjacency
    adjacency = nx.linalg.graphmatrix.adjacency_matrix(
        graph, nodelist=list_of_nodes
    )
    # make a dataframe with the features
    node_dicts = []
    for i, node in enumerate(list_of_nodes):
        node_dict = graph.nodes()[node]
        node_dict["node_id"] = node
        node_dict["id"] = i
        node_dicts.append(node_dict)
    node_feat_df = pd.DataFrame(node_dicts).set_index("id")

    # add degrees
    out_degree = np.array(np.sum(adjacency, axis=0)).flatten()
    in_degree = np.array(np.sum(adjacency, axis=1)).flatten()
    node_feat_df["in_degree"] = in_degree
    node_feat_df["out_degree"] = out_degree
    return adjacency, node_feat_df


studies = [
    "geolife"
]  # ['tist_toph100', 'tist_random100'] #, 'tist_toph10', 'tist_top100', 'tist_toph100', 'tist_top500',
# 'tist_toph500', 'tist_top1000', 'tist_toph1000']

epsg_for_study = {
    "gc1": "EPSG:21781",
    "gc2": "EPSG:21781",
    "yumuv": "EPSG:21781",
    "geolife": None,  # using haversine distance then to process coordinates
}

# limit = "where user_id > 1670"
limit = ""
single_user = False

if __name__ == "__main__":

    user_id_list, adjacency_list, node_feat_list = [], [], []
    for study in studies:
        print("--------- Start {} --------------".format(study))

        engine = get_con()

        # get appropriate projection if possible
        epsg = epsg_for_study[study]
        if epsg is not None:
            transformer = Transformer.from_crs(CRS_WGS84, epsg, always_xy=True)
            out_crs = CRS.from_epsg(epsg)
        else:
            transformer, out_crs = (None, None)

        print("CRS", transformer, out_crs)

        # Download data
        (sp, locs, trips, gap_treshold) = download_data(study, engine)

        # Iterate over users and create graphs:
        for user_id in tqdm(locs["user_id"].unique()):

            # Filter for user
            sp_user = sp[sp["user_id"] == user_id]
            if sp_user.empty:
                continue
            locs_user = locs[locs["user_id"] == user_id]

            if trips is not None:
                trips_user = trips[trips["user_id"] == user_id]
                if trips_user.empty:
                    continue
            else:
                trips_user = None

            # Generate graph
            ag = generate_graph(
                locs_user,
                sp_user,
                study,
                trips_user=trips_user,
                gap_threshold=gap_treshold,
            )
            graph = ag.G
            print(
                "activity graph size",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )

            # Preprocessing graphs:
            graph = delete_zero_edges(graph)
            graph = get_largest_component(graph)
            graph = remove_loops(graph)
            if graph.number_of_edges() < 3 or graph.number_of_edges() == 0:
                print(
                    f"zero edges or not enough nodes for {study} user {user_id}"
                )
                continue

            print(
                "size after preprocessing",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
            # Optionally: keep only important nodes
            # graph = keep_important_nodes(graph, number_of_nodes)

            # convert into adjacency and node feature df
            adjacency, node_feat_df = get_adj_and_attr(graph)
            print(
                "adjacency and feature shape",
                adjacency.shape,
                node_feat_df.shape,
            )

            # Add columns for normalized coordinates
            node_feat_df = project_normalize_coordinates(
                node_feat_df, transformer=transformer, crs=out_crs
            )

            # TODO: Add POI features based on extent

            # Append
            user_id_list.append(f"{study}_{user_id}")
            adjacency_list.append(adjacency)
            node_feat_list.append(node_feat_df)
            print(node_feat_df)
            break

    with open(os.path.join("data", f"geolife_new_data.pkl"), "wb") as outfile:
        pickle.dump((user_id_list, adjacency_list, node_feat_list), outfile)
