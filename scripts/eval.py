import numpy as np
import pandas as pd
import json

from scipy.stats import pearsonr


def get_labels_preds(res, name):
    # pred = [r[2][name]["pred"] for r in res]
    raw_pred = [r[2][name]["raw_pred"] for r in res]
    # lab = [r[0] for r in res]
    raw_lab = [r[1] for r in res]
    return raw_pred, raw_lab


def create_metrics(res, name):
    raw_pred, raw_lab = get_labels_preds(res, name)
    #     if name[:3]!="knn":
    #         print(name, "STD of pred", np.std(raw_pred))
    loss = np.mean([r[2][name]["loss"] for r in res])
    abs_err = np.mean([(r[2][name]["error"]) for r in res])
    mse = np.mean([(r[2][name]["error"]) ** 2 for r in res])
    corr = pearsonr(raw_pred, raw_lab)[0]
    bal_err = balanced_abs_err(raw_lab, raw_pred)
    return {
        "loss": loss,
        "abs_err": abs_err,
        "bal_err": bal_err,
        "mse": mse,
        "corr": corr,
    }


def balanced_abs_err(raw_lab, raw_pred, return_list=False):
    raw_pred = np.array(raw_pred)
    raw_lab = np.array(raw_lab)
    all_labs = np.unique(raw_lab)
    err_per_lab = []
    for lab in all_labs:
        preds_for_lab = raw_pred[raw_lab == lab]
        abs_err_for_lab = np.abs(preds_for_lab - lab)
        err_per_lab.append(np.mean(abs_err_for_lab))
    if return_list:
        return err_per_lab
    return np.mean(err_per_lab)


def create_results_df(path_to_res_json):
    with open(path_to_res_json, "r") as infile:
        res = json.load(infile)

    # collect results in dictionaries
    res_dicts = []

    # restrict to the labels where we have a result for all models
    nr_of_models_in_res = [len(res[i][2].keys()) for i in range(len(res))]
    model_names = res[np.argmax(nr_of_models_in_res)][2].keys()
    res = [elem for elem in res if len(elem[2].keys()) == len(model_names)]

    for name in model_names:
        res_dicts.append(create_metrics(res, name))
    df = pd.DataFrame(res_dicts, index=model_names)
    df.index.name = "model"
    df = df.reset_index().sort_values("model")
    return df, res


if __name__ == "__main__":
    res_path = "outputs/results_2023.json"
    out_path = "outputs/results_2023.csv"

    df, _ = create_results_df(res_path)
    print(df)
    print("Saved csv to ", out_path)
    df.to_csv(out_path)
