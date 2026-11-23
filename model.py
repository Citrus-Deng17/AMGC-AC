import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        return torch.cat((sc_1, sc_2), 1)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    def forward(self, emb):
        global_emb = torch.mean(emb, dim=0, keepdim=True)
        return F.normalize(global_emb, p=2, dim=1)

class DGILoss(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.discriminator = Discriminator(hidden_dim)
        self.readout = AvgReadout()

    def forward(self, embeddings, negative_embeddings):
        if embeddings.shape != negative_embeddings.shape:
            raise ValueError('Positive and negative embeddings must have the same shape')
        global_repr = self.readout(embeddings)
        logits = self.discriminator(global_repr, embeddings, negative_embeddings)
        # Discriminator returns [N, 2]: positive column, negative column.
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1
        return F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h, adj)
        return h

class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)

        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(F.softplus(x), 1e-5, 1e5)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]

class AttentionFusion(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super(AttentionFusion, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, views):
        stacked = torch.stack(views, dim=0)
        weights = self.projection(stacked)
        weights = F.softmax(weights, dim=0)
        consensus_emb = torch.sum(weights * stacked, dim=0)
        return consensus_emb

class Feature_DGI_Model(nn.Module):
    def __init__(self, vdims, nhid1, nhid2, dropout, contrastive_dim=64):
        super().__init__()
        self.view_encoders = nn.ModuleList([GCN(vdim, nhid1, nhid2, dropout) for vdim in vdims])
        self.fusion = AttentionFusion(nhid2)
        self.view_decoders = nn.ModuleList([decoder(vdim, nhid1, nhid2) for vdim in vdims])
        self.dgi_loss_fn = DGILoss(nhid2)

    def encode_views(self, view_inputs, fadj):
        view_embs = []

        for i, encoder in enumerate(self.view_encoders):
            adj_i = fadj[i] if isinstance(fadj, (list, tuple)) else fadj
            view_embs.append(encoder(view_inputs[i], adj_i))

        return self.fusion(view_embs)

    def forward(self, view_inputs, fadj, return_contrastive=True):
        consensus_emb = self.encode_views(view_inputs, fadj)

        recons_results = []
        for i, decoder_net in enumerate(self.view_decoders):
            recons_results.append(decoder_net(consensus_emb))

        means = [res[2] for res in recons_results]
        mean_fused = torch.cat(means, dim=1)

        if not return_contrastive or not self.training:
            return consensus_emb, recons_results, mean_fused

        # Same node permutation for every view, with adjacency unchanged.
        permutation = torch.randperm(consensus_emb.size(0), device=consensus_emb.device)
        devices = [consensus_emb.device.index] if consensus_emb.is_cuda else []
        # Match the tested implementation: isolate negative-branch dropout RNG.
        with torch.random.fork_rng(devices=devices):
            negative_embeddings = self.encode_views(
                [x[permutation] for x in view_inputs], fadj
            )
        # Gradients flow through both branches; only clean embeddings are decoded.
        dgi_loss = self.dgi_loss_fn(consensus_emb, negative_embeddings)

        return consensus_emb, recons_results, mean_fused, dgi_loss
