import torch
import torch.nn as nn
import torch.nn.functional as F

class EM_base_AE(nn.Module):
    def __init__(self, num_classes, dim, input_dim):
        super(EM_base_AE, self).__init__()
        self.encoder = nn.Sequential(
            # nn.Dropout(0.25),
            nn.Linear(input_dim, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 2000),
            nn.LeakyReLU(),
            nn.Linear(2000, 10),
        )
        self.decoder = nn.Sequential(
            # nn.Dropout(0.25),
            nn.Linear(dim, 2000),
            nn.LeakyReLU(),
            nn.Linear(2000, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 500),
            nn.LeakyReLU(),
            nn.Linear(500, input_dim),
        )

        self.num_classes = num_classes
        self.dim = dim
        self.Clustering = VectorQuantizerEMA(self.num_classes, self.dim)
        self.fc0 = nn.Linear(2 * self.dim, self.dim)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def Encode(self, x):
        z = self.encoder(x)
        return z

    def reparametrize(self, mu):
        eps = 0.5 * mu.data.new(mu.size()).normal_()
        return eps.add(mu)

    def Decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.Encode(x)
        quantized, gamma, centroid = self.Clustering(z)

        ## regular training
        double_z = torch.cat((z, quantized), 1)
        recon_z = self.fc0(double_z)
        recon_z = self.reparametrize(recon_z)
        recon_x = self.Decode(recon_z)

        ## Centroids reconstruction
        double_center = torch.cat((centroid, centroid), 1)
        fuse_center = self.fc0(double_center)
        recon_center = self.Decode(fuse_center)
        return z, quantized, gamma, recon_x, recon_center

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, delta=-5, decay=0.995, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.centroids = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.centroids.weight.data = torch.nn.init.xavier_uniform_(torch.empty(self._num_embeddings, self._embedding_dim))

        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_embeddings))
        self._ema_w = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self._num_embeddings, self._embedding_dim)))

        self.delta = delta
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, z):

        # Calculate distances
        Dist = torch.unsqueeze(z, 1) - self.centroids.weight
        z_dist = torch.sum(torch.mul(Dist, Dist), 2)
        gamma = F.softmax(self.delta * z_dist, dim=1)

        # Encoding
        encoding_Idx = torch.argmax(gamma, dim=1).unsqueeze(1)
        Idx = torch.zeros(z.shape[0], self._num_embeddings, device=z.device)
        Idx.scatter_(1, encoding_Idx, 1)

        # Quantize and reconstruction
        recons = torch.matmul(Idx, self.centroids.weight)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(Idx, 0)
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(Idx.t(), z)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self.centroids.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Straight Through Estimator
        recons = z + (recons - z).detach()
        cluster_center = self.centroids.weight.data

        return recons, gamma, cluster_center
