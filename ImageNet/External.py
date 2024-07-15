import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms as T
from torchvision import datasets as dset

def get_data_loader(dataset, batch_size=256, num_workers=0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def scatter(x, mu, colors, ID):
    n_color = colors.max() + 1
    palette = np.array(sns.color_palette("hls", n_color))

    f = plt.figure()
    ax = plt.subplot(aspect='equal')
    for i in range(n_color):
        positions = np.where(colors == i)
        ax.scatter(x[positions[0], 0], x[positions[0], 1], lw=0, s=8, alpha=0.3,
                   c=palette[colors[positions[0]].astype(np.int)], label='{}'.format(i))
    for i in range(mu.shape[0]):
        ax.scatter(mu[i, 0], mu[i, 1], lw=0, s=200, alpha=0.5,
                   c='grey', label='{}'.format(i))
    ax.axis('off')
    ax.axis('tight')
    #plt.legend()

    txts = []
    for i in range(n_color):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=12, alpha=0.3)
        #txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
        txts.append(txt)
    if ID == 1:
        plt.savefig('embeddings_11.png')
    elif ID == 2:
        plt.savefig('embeddings_12.png')
    elif ID == 3:
        plt.savefig('embeddings_13.png')
    else:
        plt.savefig(ID)

    #plt.show()
    plt.close(f)
    return f

def tsne_figure(feature, mu, number_label, ID):
    all_data = np.concatenate([feature, mu], axis=0)
    proj_feat = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=9159).fit_transform(all_data)
    feature = proj_feat[0:feature.shape[0],:]
    mu = proj_feat[feature.shape[0]: feature.shape[0]+mu.shape[0],:]
    scatter(feature, mu, number_label, ID)

def center_figure(centroids, ID):
    plt.gray()
    fig = plt.figure(figsize=(16,7))
    for i in range(0, centroids.shape[0]):
        #ax = fig.add_subplot(2, 5, i+1, title = "Centroid for Digit:{}".format(str( centroids.label[i] )))
        ax = fig.add_subplot(2, 5, i+1)
        ax.matshow( centroids[i,].reshape((28,28)).astype(float))
    if ID == 1:
        plt.savefig('./Results/ours/mnist_centroids.png')
    elif ID == 2:
        plt.savefig('./Results/fixed_centroid/mnist_centroids.png')
    elif ID == 3:
        plt.savefig('./Results/fixed_centroid_cos/mnist_centroids.png')
    elif ID == 4:
        plt.savefig('./Results/SGD_centroid/mnist_centroids.png')
    #plt.show()
    plt.close(fig)

def Figure_Plot(Embedding, label, centroids, ID, mu):
    tsne_figure(Embedding, mu, label, ID)
    if centroids is not None:
        center_figure(centroids, ID)

def cluster_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from scipy.optimize import linear_sum_assignment
    NMI_fun = normalized_mutual_info_score
    ARI_fun = adjusted_rand_score

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    Acc = w[row_ind, col_ind].sum() / y_pred.size
    nmi = NMI_fun(y_true, y_pred)
    ari = ARI_fun(y_true, y_pred)
    return Acc, nmi, ari