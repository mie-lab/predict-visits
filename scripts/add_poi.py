import os
import pickle


def add_poi(feature_df):
    # extract extent column
    extent = feature_df["extent"]
    # TODO: for each row in the feature dataframe, add columns with the poi
    # information (put tf-idf vector as columns) - using column "extent"

    # return dataframe with the new columns
    return feature_df


if __name__ == "__main__":
    dataset = "train_data_22"

    with open(os.path.join("data", dataset + "pkl"), "rb") as infile:
        (user_id_list, adjacency_list, node_feat_list) = pickle.load(infile)

    node_feat_list_w_poi = []
    for i in range(len(node_feat_list)):
        print(user_id_list[i])
        new_features = add_poi()

        node_feat_list_w_poi.append(new_features)

    # save new data
    with open(os.path.join("data", dataset + "_poi.pkl"), "wb") as outfile:
        pickle.dump(
            (user_id_list, adjacency_list, node_feat_list_w_poi), outfile
        )
