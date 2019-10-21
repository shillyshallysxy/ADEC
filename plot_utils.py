from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def save_scattered_image(z, id, path='./results/scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    if len(id.shape) == 2:
        c = np.argmax(id, 1)
    else:
        c = id
    if len(z[0]) == 2:
        plt.scatter(z[:, 0], z[:, 1], c=c, marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        axes = plt.gca()
        axes.set_xlim([-4.5, 4.5])
        axes.set_ylim([-4.5, 4.5])
    else:
        z_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(z)
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=c, marker='o', edgecolor='none',
                    cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    plt.grid(True)
    plt.savefig(path)
    plt.close()
