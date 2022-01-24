import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb
from tqdm import tqdm
from pathlib import Path

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel
import gensim

from scripts.graph_preprocessing import CRS_WGS84


def add_poi(feature_df, poiRep_df):

    new_feature_df = feature_df.merge(poiRep_df, on="location_id", how="left")
    new_feature_df.drop(columns=["extent"], inplace=True)

    ## we fill in all 0s for nan
    # get len of a sample normal poi vector
    na_records = new_feature_df["poiRep"].isna()
    poiVec_length = len(new_feature_df.loc[~na_records].iloc[0]["poiRep"])
    # fill in nan records
    new_feature_df["poiRep"] = new_feature_df["poiRep"].apply(
        lambda x: list(np.zeros(poiVec_length)) if np.all(pd.isna(x)) else x
    )

    assert (
        new_feature_df.dropna(subset=["poiRep"]).shape[0] == feature_df.shape[0]
    )

    # return dataframe with the new columns
    return new_feature_df


def get_loc_poi_pair(
    all_node_feat_list,
    save_path,
    buffer_distance=50,
    method="lda",
    vector_len=16,
):
    """
    return poi vector representation of each location.

    1. buffer location extent with buffer_distance
    2. get within poi of each location
    3. calculate poi vector representation using method={"lda", "tf_idf"}
    4. return poi_dict, with poi_dict["index"] the location_id and
    poi_dict["poiValues"] the poi vector.

    """
    # concat all locations
    locs = pd.concat(all_node_feat_list).drop_duplicates(subset="location_id")
    locs = locs[["location_id", "extent"]]

    # transform to gdf
    locs["extent"] = locs["extent"].apply(wkb.loads, hex=True)
    locs = gpd.GeoDataFrame(locs, geometry="extent", crs=CRS_WGS84)

    # extend the location area for a buffer distance. checked
    locs = locs.to_crs("EPSG:2056")
    locs["extent"] = locs["extent"].buffer(distance=buffer_distance)

    # read poi
    poi = gpd.read_file(os.path.join("data", "poi", "final_pois.shp"))

    # get the inside poi within each location
    tqdm.pandas(desc="pandas bar")
    spatial_index = poi.sindex
    locs["poi_within"] = locs["extent"].progress_apply(
        _get_inside_pois, poi=poi, spatial_index=spatial_index
    )

    # cleaning and expanding to location_id-poi_id pair
    locs.drop(columns="extent", inplace=True)
    # explode preserves nan - preserves locs with no poi
    locs_poi = locs.explode(column="poi_within")

    # get the poi info from original poi df
    locs_poi = locs_poi.merge(
        poi[["id", "category", "code"]],
        left_on="poi_within",
        right_on="id",
        how="left",
    )
    # here we drop locs with no poi inside
    locs_poi.drop(columns=["id"], inplace=True)

    # get the final poi and location pairs
    valid_pairs = locs_poi.dropna(subset=["poi_within"]).copy()
    valid_pairs["code"] = valid_pairs["code"].astype(int).astype(str)
    # valid_pairs.to_csv(os.path.join("data", "poi", "loc_pois_pair.csv"))

    # get the poi representation
    if method == "tf_idf":
        poi_dict = _tf_idf(valid_pairs, categories=vector_len)
    elif method == "lda":
        poi_dict = _lda(valid_pairs, categories=vector_len)
    else:
        raise AttributeError

    # save the poi representation
    with open(save_path, "wb") as handle:
        pickle.dump(poi_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return poi_dict


def _tf_idf(df, categories=8):
    """
    tf_idf method to get the poi representation of each location.

    categories is the length of the output vector representation
    for each location.
    """
    texts = df.groupby("location_id")["category"].apply(list).to_list()

    dct = Dictionary(texts)
    corpus = [dct.doc2bow(line) for line in texts]

    tfmodel = TfidfModel(corpus)
    vector = tfmodel[corpus]

    # the tf array
    dense_tfvector = gensim.matutils.corpus2dense(
        vector, num_terms=categories
    ).T
    # the index arr
    index_arr = (
        df.groupby("location_id").count().reset_index()["location_id"].values
    )
    return {"index": index_arr, "poiValues": dense_tfvector}


def _lda(df, categories=16):
    """
    lda method to get the poi representation of each location.

    categories is the length of the output vector representation
    for each location.
    """
    texts = df.groupby("location_id")["code"].apply(list).to_list()

    dct = Dictionary(texts)
    corpus = [dct.doc2bow(line) for line in texts]

    lda = LdaModel(corpus, num_topics=categories)
    vector = lda[corpus]

    # the lda array
    dense_ldavector = gensim.matutils.corpus2dense(
        vector, num_terms=categories
    ).T
    # the index arr
    index_arr = (
        df.groupby("location_id").count().reset_index()["location_id"].values
    )
    return {"index": index_arr, "poiValues": dense_ldavector}


def _get_inside_pois(df, poi, spatial_index):
    """
    Given one extent (df), return the poi within this extent.

    spatial_index is obtained from poi.sindex to speed up the process.
    """
    possible_matches_index = list(spatial_index.intersection(df.bounds))
    possible_matches = poi.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.within(df)]["id"].values

    return precise_matches


if __name__ == "__main__":
    dataset = "new"

    with open(os.path.join("data", dataset + ".pkl"), "rb") as infile:
        (user_id_list, adjacency_list, node_feat_list) = pickle.load(infile)

    # the get_loc_poi_pair is slow:
    # if the file already exist we directly read from file
    poiPath = os.path.join("data", f"{dataset}_poiRep.pk")
    if Path(poiPath).is_file():
        poiRep = pickle.load(open(poiPath, "rb"))
    else:  # or we generate with the function. SLOW!!
        poiRep = get_loc_poi_pair(
            node_feat_list,
            buffer_distance=50,
            method="lda",
            vector_len=16,
            save_path=poiPath,
        )
    # the dict is transformed to df
    poiRep_df = pd.DataFrame(poiRep["index"], columns=["location_id"])
    poiRep_df["poiRep"] = poiRep["poiValues"].tolist()

    node_feat_list_w_poi = []
    for i in range(len(node_feat_list)):
        new_features = add_poi(node_feat_list[i], poiRep_df)

        node_feat_list_w_poi.append(new_features)

    # save new data
    with open(os.path.join("data", dataset + "_poi.pkl"), "wb") as outfile:
        pickle.dump(
            (user_id_list, adjacency_list, node_feat_list_w_poi), outfile
        )
