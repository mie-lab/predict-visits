import os
import pickle
import numpy as np
import networkx as nx
import pandas as pd
from pyproj import Transformer, CRS
import copy
import json
import geopandas as gpd
import psycopg2
from tqdm import tqdm
from functools import wraps
from shapely import wkt, wkb
import argparse

import trackintel as ti
from graph_trackintel.activity_graph import ActivityGraph
from graph_trackintel.graph_utils import (
    delete_zero_edges,
    get_largest_component,
    remove_loops,
    get_adj_and_attr,
)


CRS_WGS84 = "epsg:4326"

epsg_for_study = {
    "gc1": 21781,
    "gc2": 21781,
    "yumuv": 21781,
}
has_trip_dict = {
    "gc1": True,
    "gc2": True,
    "yumuv_graph_rep": True,
    "geolife": True,
}
sp_name_dict = {
    "gc1": "staypoints_extent",
    "gc2": "staypoints_extent",
    "yumuv_graph_rep": "staypoints",
    "geolife": "staypoints_extent",
    "tist_toph100": "staypoints_extent",
    "tist_random100": "staypoints",
    "tist_toph1000": "staypoints",
}
locs_name_dict = {
    "gc1": "locations_extent",
    "gc2": "locations_extent",
    "yumuv_graph_rep": "locations",
    "geolife": "locations_extent",
    "tist_toph100": "locations_extent",
    "tist_random100": "locations",
    "tist_toph1000": "locations",
}


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


def download_data(study, engine, has_trips=True, filter_coverage=False):
    """Download data of one study from database"""

    def to_datetime(df):
        df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
        df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
        return df

    exclude_purpose_tist = [
        "Light Rail",
        "Subway",
        "Platform",
        "Trail",
        "Road",
        "Train",
        "Bus Line",
    ]

    print("\t download staypoints")
    sp = gpd.read_postgis(
        sql="select * from {}.{}".format(study, sp_name_dict[study]),
        con=engine,
        geom_col="geom",
        index_col="id",
    )
    sp = to_datetime(sp)

    print("\t download locations")
    sql = f"SELECT * FROM {study}.{locs_name_dict[study]}"
    locs = ti.io.read_locations_postgis(sql, con=engine, index_col="id")

    # studies with trips
    gap_treshold = 12
    if has_trips:
        print("\t download trips")
        trips = ti.io.read_trips_postgis(
            f"select * from {study}.trips", con=engine
        )
        if filter_coverage:
            # get_triplegs(study=study, engine=engine)
            print("\t download triplegs")
            tpls = pd.read_sql(
                sql=f"select id, user_id, started_at, finished_at from {study}.triplegs",
                con=engine,
                index_col="id",
            )
            tpls = to_datetime(tpls)
            sp, locs, trips = filter_tracking_coverage(sp, locs, tpls, trips)

    # studies without trips (Foursquare)
    else:
        sp = sp[~sp["purpose"].isin(exclude_purpose_tist)]
        trips = None
    return (sp, locs, trips, gap_treshold)


def filter_tracking_coverage(sp, locs, tpls, trips):
    """Filter out users where tracking coverage is too low"""
    print("\t filter by tracking coverage")

    def filter_user_by_number_of_days(
        sp, tpls, coverage=0.7, min_nb_good_days=28, filter_sp=True
    ):
        nb_users = len(sp.user_id.unique())

        sp_tpls = sp.append(tpls).sort_values(["user_id", "started_at"])

        coverage_df = ti.analysis.tracking_quality.temporal_tracking_quality(
            sp_tpls, granularity="day", max_iter=1000
        )

        good_days_count = (
            coverage_df[coverage_df["quality"] >= coverage]
            .groupby(by="user_id")["quality"]
            .count()
        )
        good_users = good_days_count[good_days_count >= min_nb_good_days].index
        if filter_sp:
            sp = sp[sp.user_id.isin(good_users)]
            print(
                "\t\t nb users now: ",
                len(sp.user_id.unique()),
                "before: ",
                nb_users,
            )
        return sp, good_users

    if study == "geolife":
        sp, user_id_ix = filter_user_by_number_of_days(
            sp=sp, tpls=tpls, coverage=0.7, min_nb_good_days=14
        )
    else:
        sp, user_id_ix = filter_user_by_number_of_days(
            sp=sp, tpls=tpls, coverage=0.7, min_nb_good_days=14
        )
    print("\t\t drop users with bad coverage: before: ", len(locs))
    tpls = tpls[tpls.user_id.isin(user_id_ix)]
    trips = trips[trips.user_id.isin(user_id_ix)]
    locs = locs[locs.user_id.isin(user_id_ix)]
    print("after:", len(locs))
    return sp, locs, trips


def generate_graph(
    locs_user, sp_user, study, trips_user=None, gap_threshold=None
):
    """
    Given the locations and staypoints OF ONE USER, generate the graph
    """
    # print(sp_user["location_id"])
    # print(locs_user)
    AG = ActivityGraph(
        sp_user,
        locs_user,
        trips=trips_user,
        gap_threshold=gap_threshold,
    )
    # Add purpose feature
    if study == "geolife" or study == "yumuv_graph_rep":
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

    @wraps(func)
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
            home_x, home_y = transformer.transform(home_center.x, home_center.y)
            return (proj_x - home_x, proj_y - home_y)
        else:  # fall back to haversine
            return get_haversine_displacement.__wrapped__(x, y, home_center)

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

    # add_distance
    normed_coords["distance"] = normed_coords.apply(
        lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2), axis=1
    )
    # TODO: add as a TEST! compare haversine dist to fake-projected coordinates
    # print(normed_coords["distance"])
    # test_distance = node_feats["center"].apply(
    #     lambda point: ti.geogr.point_distances.haversine_dist(
    #         point.x, point.y, home_center.x, home_center.y
    #     )[0]
    # )
    # print(
    #     pd.merge(
    #         normed_coords, test_distance, left_index=True, right_index=True
    #     )
    # )

    return pd.merge(
        node_feats, normed_coords, left_index=True, right_index=True
    )


def filter_for_time_period(
    sp_user, locs_user, part_start_date, part_end_date, trips_user=None
):
    # filter for current time period --> use trips if available to filter time
    if trips_user is not None:
        trips_user_k = trips_user[
            (trips_user["started_at"] < part_end_date)
            & (trips_user["started_at"] >= part_start_date)
        ]
        # filter staypoints by trips because we later form the graph based on the trips!
        sp_user_k = sp_user[
            (sp_user.index.isin(trips_user_k["origin_staypoint_id"]))
            | (sp_user.index.isin(trips_user_k["destination_staypoint_id"]))
        ]

    else:
        trips_user_k = None
        sp_user_k = sorted_sp[
            (sorted_sp["started_at"] < part_end_date)
            & (sorted_sp["started_at"] >= part_start_date)
        ]
    locs_user_k = locs_user[locs_user.index.isin(sp_user_k["location_id"])]

    return sp_user_k, locs_user_k, trips_user_k


def getmost(val_list):
    """Helper function to get the value that appears most often in a list"""
    # Problem: In GC datasets is the purpose a string-list
    if val_list[0][0] == "[":
        val_list = [elem for val in val_list for elem in eval(val)]
    # remove unkowns
    val_list = [elem for elem in val_list if elem != "unknown"]
    if len(val_list) == 0:
        return "unknown"
    # Select the most prevalent purpose
    uni, counts = np.unique(val_list, return_counts=True)
    return uni[np.argmax(counts)]


def average_hour(datetime_list):
    hours = [t.hour for t in datetime_list]
    return np.mean(hours)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--node_thresh",
        type=int,
        default=30,
        help="Minimum number of nodes to include the graph",
    )
    parser.add_argument(
        "-s",
        "--save_name",
        type=str,
        default="new",
        help="Name under which to save the data",
    )
    parser.add_argument(
        "-t",
        "--time_period",
        type=int,
        default=600,
        help="Time period for one graph (in number of days)",
    )
    parser.add_argument(
        "-b",
        "--one_bin_per_user",
        type=int,
        default=1,
        help="set to 0 if one user can appear multiple times",
    )
    args = parser.parse_args()

    min_nodes = args.node_thresh
    save_name = args.save_name

    studies = ["tist_toph1000" "gc1", "gc2", "yumuv_graph_rep", "geolife"]

    for study in studies:
        print("--------- Start {} --------------".format(study))

        engine = get_con()

        # get appropriate projection if possible
        # If None, using haversine distance to process coordinates
        epsg = epsg_for_study.get(study, None)
        if epsg is not None:
            transformer = Transformer.from_crs(CRS_WGS84, epsg, always_xy=True)
            out_crs = CRS.from_epsg(epsg)
        else:
            transformer, out_crs = (None, None)

        print("CRS", transformer, out_crs)

        # Download data
        (sp, locs, trips, gap_treshold) = download_data(
            study, engine, has_trips=has_trip_dict.get(study, False)
        )

        user_id_list, adjacency_list, node_feat_list = [], [], []
        # Iterate over users and create graphs:
        for user_id in tqdm(locs["user_id"].unique()):
            print("-------------------------")
            print(user_id)

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

            # divide into time bin pieces
            sorted_sp = sp_user.sort_values("started_at")

            min_date = sorted_sp.iloc[0].loc["started_at"]
            max_date = sorted_sp.iloc[-1].loc["started_at"]
            time_period = pd.Timedelta("{} d".format(args.time_period))
            nr_parts = (max_date - min_date).days // args.time_period + 1
            # print(user_id, min_date, max_date, nr_parts)
            # cutoff = int(len(sorted_sp) / nr_parts)

            for k in range(nr_parts):
                # Restrict to this time period (k-th time bin)
                part_start_date, part_end_date = (
                    min_date + k * time_period,
                    min_date + (k + 1) * time_period,
                )
                sp_user_k, locs_user_k, trips_user_k = filter_for_time_period(
                    sp_user,
                    locs_user,
                    part_start_date,
                    part_end_date,
                    trips_user=trips_user,
                )
                if len(sp_user_k) == 0:
                    print("Warning: zero staypoints in time bin, continue")
                    continue
                # print(
                #     "start and end date",
                #     sp_user_k.iloc[0].loc["started_at"],
                #     sp_user_k.iloc[-1].loc["started_at"],
                # )

                # Generate graph
                ag = generate_graph(
                    locs_user_k,
                    sp_user_k,
                    study,
                    trips_user=trips_user_k,
                    gap_threshold=gap_treshold,
                )
                assert ag.user_id == user_id
                graph = ag.G
                # print(
                #     "activity graph size",
                #     graph.number_of_nodes(),
                #     graph.number_of_edges(),
                # )

                # Preprocessing graphs:
                graph = delete_zero_edges(graph)
                if graph.number_of_nodes() < min_nodes:
                    print(
                        f"zero edges or not enough nodes for {study} user {user_id}"
                    )
                    continue
                graph = get_largest_component(graph)
                graph = remove_loops(graph)
                if (
                    graph.number_of_nodes() < min_nodes
                    or graph.number_of_edges() == 0
                ):
                    print(
                        f"zero edges or not enough nodes for {study} user {user_id}"
                    )
                    continue

                # print(
                #     "size after preprocessing",
                #     graph.number_of_nodes(),
                #     graph.number_of_edges(),
                # )
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
                # preprocess purpose and started_at features
                if "purpose" in node_feat_df.columns:
                    node_feat_df["purpose"] = node_feat_df["purpose"].apply(
                        getmost
                    )
                node_feat_df["started_at"] = node_feat_df["started_at"].apply(
                    average_hour
                )
                node_feat_df.drop("finished_at", axis=1, inplace=True)

                # Append
                user_id_list.append(f"{study}_{user_id}_{k}")
                adjacency_list.append(adjacency)
                node_feat_list.append(node_feat_df)
                # break after we found one working time period for the user:
                if args.one_bin_per_user:
                    break

        with open(
            os.path.join("data", f"{save_name}_{study}.pkl"), "wb"
        ) as outfile:
            pickle.dump((user_id_list, adjacency_list, node_feat_list), outfile)
