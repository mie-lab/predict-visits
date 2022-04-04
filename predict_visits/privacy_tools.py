from json import load
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch_geometric.data import DataLoader

from predict_visits.dataset import MobilityGraphDataset
from predict_visits.utils import load_model

max_age = 80
min_age = 20
post_functions = {
    "income": lambda x: 9 * x,
    "age": lambda x: x * (max_age - min_age) + min_age,
    "sex": lambda x: x,
}


def retrieve_users(user_id_list):
    user_ids = [user.split("_")[-2] for user in user_id_list]
    study = ["_".join(user.split("_")[:-2]) for user in user_id_list]
    return study, user_ids


def evaluate_privacy(eval_model, test_data, task="income", return_mode="mean"):
    post_transform = post_functions[task]
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    sum_err, sum_mse = 0, 0
    res_dict = []
    for k, data in enumerate(test_data_loader):
        lab = data.y[:, -1].clone()
        # forward pass
        out = eval_model(data)

        test_mse = torch.sum((out - lab) ** 2).item()
        test_abs_err = torch.abs(
            post_transform(out) - post_transform(lab)
        ).item()
        sum_err += test_abs_err
        sum_mse += test_mse
        res_dict.append(
            {
                "user_id": data.user_id,
                "lab_normed": lab.item(),
                "lab": post_transform(lab).item(),
                "pred_normed": out.item(),
                "pred": post_transform(out).item(),
            }
        )
    res_df = pd.DataFrame(res_dict)

    print("average mse:", sum_mse / k)
    if return_mode == "mean":
        return sum_err / k
    else:
        return res_df


class PrivacyDataset(MobilityGraphDataset):
    def __init__(
        self, *args, predict_variable="income", mode="train", **kwargs
    ):
        super().__init__(*args, **kwargs)

        # load data from server
        study, user_ids = retrieve_users(self.users)
        privacy_labels = pd.read_csv(os.path.join("data", "privacy.csv"))
        privacy_labels["user_id"] = privacy_labels["user_id"].astype(str)
        privacy_labels.set_index("user_id", inplace=True)

        # select train / test set
        np.random.seed(3)
        rand_ids = np.random.permutation(privacy_labels.index)
        if mode == "train":
            use_ids = rand_ids[len(privacy_labels) // 10 :]
        else:
            use_ids = rand_ids[: len(privacy_labels) // 10]

        self.valid_samples = []
        self.labels = []
        for i in range(len(self.users)):
            user_id = user_ids[i]
            try:
                lab = privacy_labels.loc[user_id][predict_variable]

                assert pd.isna(lab) or 0 <= lab <= 1
            except KeyError:
                lab = pd.NA
            if not pd.isna(lab) and user_id in use_ids:
                self.valid_samples.append(i)
            self.labels.append(lab)
        print(
            "Valid samples", len(self.valid_samples), "out of", len(self.users)
        )

    def len(self):
        return len(self.valid_samples)

    def get(self, index):
        # transform index to use only valid samples
        idx = self.valid_samples[index]

        adj = self.adjacency[idx]
        node_feat = self.node_feats[idx]
        label = self.labels[idx]
        user_id = self.users[idx]

        predict_node_feats = np.zeros(node_feat.shape[1])
        predict_node_feats[-1] = label

        # transform to pytorch geometric data
        data_sample = self.transform_to_torch(
            adj,
            node_feat,
            predict_node_feats,
            relative_feats=False,
            user_id=user_id,
        ).to(self.device)
        return data_sample


def test_dataset_class():
    dataset = PrivacyDataset(["t120_gc1_poi.pkl"])
    test = dataset[0]
    print(test.x.size())
    print(test.y.size())
    print(test.edge_index.size())
    print(test.edge_attr.size())
    import torch_geometric

    loader = torch_geometric.loader.DataLoader(
        dataset, batch_size=2, shuffle=True
    )
    print("loader", len(loader))
    print("loader ds size", len(loader.dataset))

    for test in loader:
        print(test.x.size())
        print(test.y.size())
        print(test.y)
        break


if __name__ == "__main__":
    # test_dataset_class()

    model_path = "income_ff"
    data_path = ["t120_gc1_poi.pkl"]
    predict_variable = "income"

    model, cfg = load_model(model_path, use_best=True)
    dataset = PrivacyDataset(data_path, predict_variable=predict_variable)
    res_df = evaluate_privacy(
        model, dataset, predict_variable, return_mode="df"
    )
    res_df.to_csv("outputs/income_results.csv")
