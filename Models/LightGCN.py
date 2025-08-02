# Models/LightGCN.py
import torch
from torch import nn
from torch_geometric.nn import LGConv

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, K=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K

        # 사용자/아이템 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # LightGCN 레이어
        self.convs = nn.ModuleList([LGConv() for _ in range(K)])

    def forward(self, edge_index):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        # K-step 전파
        out = x
        for conv in self.convs:
            x = conv(x, edge_index)
            out += x
        out /= (self.K + 1)

        user_emb, item_emb = torch.split(out, [self.num_users, self.num_items])
        return user_emb, item_emb

    def predict_matrix(self):
        """전체 유저-아이템 스코어 행렬 계산"""
        user_emb, item_emb = self.forward(self.edge_index)
        return torch.matmul(user_emb, item_emb.t())