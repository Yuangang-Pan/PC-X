import numpy as np
import random
import torch
from External import Figure_Plot, cluster_accuracy
import os
from sklearn.cluster import KMeans
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
flag_cuda = torch.cuda.is_available()

SEED = 9159
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministics = True
torch.backends.cudnn.benchmark = False


num_cluster = 10
learning_rate = 5 * 1e-4
dim = 10
iter = 701
batch_size = 512
feat_xs = np.load('./ImageNet/imagenet_moco_feats_resnet34.npy')
labels = np.load('./ImageNet/imagenet_labels_resnet34.npy')
print(feat_xs.shape, labels.shape)
dataloader = [feat_xs, labels]
input_dim = feat_xs.shape[1]
# kmeans = KMeans(n_clusters=10, n_init=40, random_state=0, max_iter=5000, algorithm='full')
# predicted = kmeans.fit_predict(feat_xs)  # predict centers from features.
# acc, nmi, ari = cluster_accuracy(predicted, labels)
# print(acc, nmi, ari)
# cluster_centers = kmeans.cluster_centers_
# Figure_Plot(feat_xs, labels, None, 2, cluster_centers)

from main_base_EM_ae import GMM_base_AE
Embedding, label, recon_center, centroid = GMM_base_AE(flag_cuda, num_cluster, learning_rate, \
    dim, dataloader, batch_size, input_dim).process(iter)
Figure_Plot(Embedding, label, None, 3, centroid)

# from main_fixed_centroid import GMM_base_AE
# Embedding, label, recon_center = GMM_base_AE(flag_cuda, num_cluster, learning_rate, dim, dataloader).process(iter)
# Figure_Plot(Embedding, label, recon_center, 2)

# iter = 201
# from main_fixed_centroid_cos import GMM_base_AE
# Embedding, label, recon_center = GMM_base_AE(flag_cuda, num_cluster, learning_rate, dim, dataloader).process(iter)
# Figure_Plot(Embedding, label, recon_center, 3)

# iter = 1001
# from main_SGD_centroid import GMM_base_AE
# Embedding, label, recon_center = GMM_base_AE(flag_cuda, num_cluster, learning_rate, dim, dataloader).process(iter)
# Figure_Plot(Embedding, label, recon_center, 4)
