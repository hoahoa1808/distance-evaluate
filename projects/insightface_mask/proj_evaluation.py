import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config


def _eval_chunk(chunk, threshs, embs1, idx1, embs2, idx2):
    tp_fp_fn = np.empty((threshs.shape[0], 3), dtype=np.int32)

    embs_idx1 = np.vectorize(idx1.get)(chunk["item1"].values)
    embs_idx2 = np.vectorize(idx2.get)(chunk["item2"].values)
    dis = np.linalg.norm(embs1[embs_idx1] - embs2[embs_idx2], axis=1).astype(np.float32)
    labels = chunk['label'].values.astype(np.int32)

    for i, thresh in enumerate(threshs):
        preds = np.where(dis<thresh, 1, 0)
        tp_fp_fn[i, 0] = ((preds == 1) & (labels == 1)).sum()
        tp_fp_fn[i, 1] = ((preds == 1) & (labels == 0)).sum()
        tp_fp_fn[i, 2] = ((preds == 0) & (labels == 1)).sum()

    return tp_fp_fn


def eval(config, total_pairs, embs1, idx1, embs2, idx2):
    threshs = np.linspace(0., 1.5, 151)
    # print("INFO:(eval) Loading vector db...")
    # db = h5py.File(config.embs_h5)
    # embs = db['embs'][:]
    # idx = db['ids'][:]
    # idx = dict(zip(idx, list(range(idx.shape[0]))))
    # db.close()

    chunksize = 100000
    tp_fp_fn = np.zeros((threshs.shape[0], 3), dtype=np.int32)
    df = pd.read_csv(config.pairs_csv, chunksize=chunksize, iterator=True)
    print("INFO:(eval) Evaluating...")
    for chunk in tqdm(df, total=(total_pairs//chunksize + 1)):
        tp_fp_fn += _eval_chunk(chunk, threshs, embs1, idx1, embs2, idx2)

    precisions = tp_fp_fn[:, 0] / (tp_fp_fn[:, 0] + tp_fp_fn[:, 1])
    recalls = tp_fp_fn[:, 0] / (tp_fp_fn[:, 0] + tp_fp_fn[:, 2])

    fig, ax = plt.subplots()
    ax.plot(threshs, precisions, label="precision")
    ax.plot(threshs, recalls, label="recall")
    ax.set(xlabel='thresh', ylabel='score (euclidean)')
    ax.grid()
    plt.legend()
    plt.rcParams['figure.figsize'] = [9, 6]
    plt.figure(figsize=(9, 6))
    fig.savefig(config.result)


if __name__ == '__main__':
    import torch

    config_path = "/research/classification/distance_evaluation/config.yml"
    config = Config(config_path)

    cache = torch.load(config.cache_path)
    eval(config, cache["total_pairs"])