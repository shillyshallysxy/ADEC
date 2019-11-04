from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def reshape_c(id_):
    """
    :param id_: z空间编码对应的label，shape=[N, 1] 或 shape=[N, ] 或 shape=[N, n_cluster](即 one-hot)
    :return: shape=[N, ]
    """
    if len(id_.shape) == 2:
        if id_.shape[1] == 1:
            c = np.reshape(id_, [-1])
        else:
            c = np.argmax(id_, 1)
    elif len(id_.shape) == 1:
        c = id_
    else:
        raise ValueError("label不支持该Shape的输入: {}".format(id_.shape))
    return c


def save_scattered_image(z, id_, path='./results/scattered_image.jpg', cmp=None):
    """
    :param z:  z空间编码，shape=[N, feature_dims]
    :param id_:  z空间编码对应的label，shape=[N, 1] 或 shape=[N, ] 或 shape=[N, n_cluster](即 one-hot)
    :param path:  保存的图片存放的路径
    :param cmp:  z空间编码对应的预测label，shape=[N, 1] 或 shape=[N, ] 或 shape=[N, n_cluster](即 one-hot)
    :return:
    """
    N = 10
    plt.figure(figsize=(8, 6))
    c = reshape_c(id_)
    if len(z[0]) != 2:
        z = TSNE(n_components=2, learning_rate=100).fit_transform(z)
    plt.scatter(z[:, 0], z[:, 1], c=c, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    plt.grid(True)
    plt.savefig(path)
    plt.close()

    if cmp is not None:
        c = reshape_c(cmp)
        plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], c=c, marker='o', edgecolor='none',
                    cmap=discrete_cmap(N, 'jet'))
        plt.colorbar(ticks=range(N))
        plt.grid(True)
        temp_path = path.rpartition(".")
        temp_path = "".join((temp_path[0], "_pred", temp_path[1], temp_path[2]))
        plt.savefig(temp_path)
        plt.close()


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    num_ind = [w[i, j] for i, j in zip(*ind)]
    return sum(num_ind) * 1.0 / y_pred.size