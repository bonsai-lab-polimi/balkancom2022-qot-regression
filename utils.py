import ast
import os

import numpy as np
import pandas as pd


def load_csv_from_folder(path: str, rng: np.random.Generator):
    X = []
    X_mean = []
    X_sample = []
    y_mean = []
    y_std = []
    y_true = []
    y_quantiles = []
    y_samples = []
    for file in os.listdir(path):
        if file.startswith('.'):
            continue
        data = pd.read_csv(path + file, sep=",")
        X_file, X_mean_file, X_sample_file, y_mean_file, y_std_file, y_quantiles_file, y_file, y_samples_file = preprocess_data(data, rng)
        X.append(X_file)
        X_mean.append(X_mean_file)
        X_sample.append(X_sample_file)
        y_mean.append(y_mean_file)
        y_std.append(y_std_file)
        y_quantiles.append(y_quantiles_file)
        y_true.append(y_file)
        y_samples.append(y_samples_file)
    X = np.concatenate(X)
    X_mean = np.concatenate(X_mean)
    X_sample = np.concatenate(X_sample)
    y_mean = np.concatenate(y_mean)
    y_std = np.concatenate(y_std)
    y_quantiles = np.concatenate(y_quantiles)
    y_true = np.concatenate(y_true)
    y_samples = np.concatenate(y_samples)
    return X, X_mean, X_sample, y_mean, y_std, y_quantiles, y_true, y_samples

def preprocess_data(data: pd.DataFrame, rng: np.random.Generator):
    idx_requests = data["instance"].to_numpy()
    idx_paths = data["path_id"].to_numpy()
    y = data["OSNR"].to_numpy()
    mfs = data["constellation"].to_numpy()
    unique_requests = np.unique(idx_requests)
    unique_paths = np.unique(idx_paths)
    mf_list = np.unique(mfs)
    X_mean = []
    X_sample = []
    y_mean = []
    y_std = []
    y_quintiles = []
    y_samples = []

    data["link_len"] = data["link_len"].apply(ast.literal_eval)
    data["min_link"] = data["link_len"].apply(min)
    data["max_link"] = data["link_len"].apply(max)
    data["len_path"] = data["link_len"].apply(sum)
    data["hops_path"] = data["link_len"].apply(len)
    cols_to_drop = ["OSNR", "instance", "node_dst", "node_src", "amplifier_spacing", "system_margin", "fiber_loss", "ampl_noise",
                    "link_len", "hops_id", "guardband_l", "guardband_r", "traffic_l", "traffic_r", "constellation_l",
                    "constellation_r", "path_id"]
    for request in unique_requests:
        for path in unique_paths:
            for mf in mf_list:
                idx_mf_path = (request == idx_requests) * (mf == mfs) * (idx_paths == path)
                y_mf = y[idx_mf_path]
                y_mean.append(np.mean(y_mf))
                y_std.append(np.std(y_mf))
                y_quintiles.append(np.quantile(y_mf, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], method='median_unbiased'))
                X_mf = data.loc[idx_mf_path]
                X_mf = X_mf.drop(columns=cols_to_drop, inplace=False)
                X_mean.append(X_mf.to_numpy()[0])
    idxs_keep = []
    for request in unique_requests:
        mf_keep = rng.choice(mf_list)
        path_keep = rng.choice(unique_paths)
        idx_keep = (request == idx_requests) * (mf_keep == mfs) * (path_keep == idx_paths)
        idxs_keep.append(idx_keep)
    keep_bool = np.sum(np.array(idxs_keep), axis=0, dtype=np.bool8)
    X_sample = data.loc[keep_bool].drop(columns=cols_to_drop, inplace=False).to_numpy()
    y_samples = y[keep_bool]
    data = data.drop(columns=cols_to_drop)
    X = data.to_numpy()
    return X, np.array(X_mean), X_sample, np.array(y_mean), np.array(y_std), np.array(y_quintiles), y, y_samples

def get_lightpath(data: pd.DataFrame, idx_pred: int):
    idx_request = data["instance"].to_numpy()[idx_pred]
    idx_path = data["path_id"].to_numpy()[idx_pred]
    mf = data["constellation"].to_numpy()[idx_pred]
    y = data["OSNR"].to_numpy()
    idx_mf_path = (data["instance"].to_numpy() == idx_request) * (data["constellation"].to_numpy() == mf) * (data["path_id"].to_numpy() == idx_path)
    y_mf = y[idx_mf_path]
    data["link_len"] = data["link_len"].apply(ast.literal_eval)
    data["min_link"] = data["link_len"].apply(min)
    data["max_link"] = data["link_len"].apply(max)
    data["len_path"] = data["link_len"].apply(sum)
    data["hops_path"] = data["link_len"].apply(len)
    cols_to_drop = ["OSNR", "instance", "node_dst", "node_src", "amplifier_spacing", "system_margin", "fiber_loss", "ampl_noise",
                "link_len", "hops_id", "guardband_l", "guardband_r", "traffic_l", "traffic_r", "constellation_l",
                "constellation_r", "path_id"]
    data = data.drop(columns=cols_to_drop)
    x = data.iloc[idx_pred].to_numpy()
    return x, y_mf
