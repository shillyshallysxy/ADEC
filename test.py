# -*- encoding:utf8 -*-
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import plot_utils as pu

# z = np.load("./results/z.npy")[:10000, :]
# z = TSNE(n_components=2, learning_rate=100).fit_transform(z)
# z = z[:, :2]
# print(z.shape)
# y = np.load("./results/label.npy")[:10000]
#
# # y_pred = SpectralClustering(n_clusters=10).fit_predict(z)
# # y_pred = KMeans(n_clusters=10, max_iter=10).fit_predict(z)
# y_pred = MiniBatchKMeans(n_clusters=10, batch_size=256, n_init=20).fit_predict(z)
# print("聚类完毕")
#
# pu.save_scattered_image(z, y, "./results/z_spectral.jpg", y_pred)

def combinationSum(candidates, target):
    res = list()

    def fetch_res(candidates, now_res, target):
        if target == 0:
            res.append(now_res)
        elif target < 0 or len(candidates) == 0:
            return
        else:
            fetch_res(candidates, now_res + [candidates[0]], target - candidates[0])
            fetch_res(candidates[1:], now_res, target)

    fetch_res(candidates, [], target)
    return res

# import nltk
# nltk.download()
import sqlite3
# import site
# print(site.tsitepackages())