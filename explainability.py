"""Explaining models for CTD"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def l0_hc_loss(u, alpha=0.5, beta=0.8, gamma=-0.2, zeta=1.2):
    u = torch.clamp_min(u, 1e-18)
    return torch.sigmoid((torch.log(u) - beta * math.log(-gamma / zeta))).mean()


class CTDExplainer(nn.Module):
    def __init__(self, gae_model, target_edge_index):
        """Initializes the explainer for the model

        Parameters
        ----------
        gae_model : a class that implements PtG GAE interface
            The number of channels in the input graph nodes
        target_edge_index : torch.LongTensor
            The edges to explain
        """
        super(CTDExplainer, self).__init__()
        self.model = gae_model
        self.target_edge_index = target_edge_index
        self.bilin = nn.parameter.Parameter(
            torch.rand(
                (3, self.model.encoder.in_channels, self.model.encoder.in_channels)
            ),
            requires_grad=True,
        )

    def forward(self, x, adj_t, edge_types):
        """Calculates embeddings and returns importance masks"""
        H = torch.cat((x, self.model.encoder.emb), 0)
        z0 = H @ self.bilin[0] @ H.T
        a0 = adj_t.storage.set_value(
            z0[adj_t.storage.row(), adj_t.storage.col()] * adj_t.storage.value(),
            layout="coo",
        )
        H = self.model.encoder.norm1(self.model.encoder.conv1(H, a0, edge_types))

        z1 = H @ self.bilin[1] @ H.T
        a1 = adj_t.storage.set_value(
            z1[adj_t.storage.row(), adj_t.storage.col()] * adj_t.storage.value(),
            layout="coo",
        )

        H = self.model.encoder.norm2(
            self.model.encoder.conv2(F.relu(H), a1, edge_types)
        )

        z2 = H @ self.bilin[2] @ H.T
        a3 = adj_t.storage.set_value(
            z2[adj_t.storage.row(), adj_t.storage.col()] * adj_t.storage.value(),
            layout="coo",
        )
        H = self.model.encoder.conv3(F.relu(H), a3, edge_types)
        return H, torch.stack((z0, z1, z2))

    def calculate_loss(self, x, adj_t, edge_types, l0_reg=0.002):
        H = self.forward(x, adj_t, edge_types)
        res_sp = self.model.decode(H, self.target_edge_index)

        H, Z = self.model.encode(x, adj_t, edge_types)
        res = self.model.decode(H, self.target_edge_index)
        return ((res - res_sp) ** 2).mean() + l0_reg * l0_hc_loss(Z)
