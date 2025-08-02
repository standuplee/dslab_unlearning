import torch
from torch import nn, optim, Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import structured_negative_sampling
from torch_sparse import SparseTensor, matmul
import numpy as np
import random
import time


class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = [emb_0]
        emb_k = emb_0

        for _ in range(self.K):
            emb_k = self.propagate(edge_index, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)

        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items])
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)

    def train_model(self, train_edge_index, test_edge_index, active_edge_index=None, inactive_edge_index=None,
                    epochs=200, batch=1024, lr=1e-3, per_eval=50, per_lr_decay=200, K=20, LAMBDA=1e-6, verbose=2):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")


        self.to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        best_ndcg, best_hr = 0, 0
        active_ndcg, inactive_ndcg = 0, 0

        train_edge_index = train_edge_index.to(device)
        test_edge_index = test_edge_index.to(device)
        if active_edge_index is not None:
            active_edge_index = active_edge_index.to(device)
        if inactive_edge_index is not None:
            inactive_edge_index = inactive_edge_index.to(device)

        start_time = time.time()

        for it in range(epochs):
            self.train()
            users_emb_final, users_emb_0, items_emb_final, items_emb_0 = self.forward(train_edge_index)

            user_indices, pos_item_indices, neg_item_indices = sample_batch(batch, train_edge_index)
            user_indices, pos_item_indices, neg_item_indices = user_indices.to(device), pos_item_indices.to(device), neg_item_indices.to(device)

            loss = bpr_loss(users_emb_final[user_indices], users_emb_0[user_indices],
                            items_emb_final[pos_item_indices], items_emb_0[pos_item_indices],
                            items_emb_final[neg_item_indices], items_emb_0[neg_item_indices], LAMBDA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % per_eval == 0:
                self.eval()
                val_loss, ndcg, hr = evaluation(self, test_edge_index, test_edge_index, [train_edge_index], K, LAMBDA)
                if verbose >= 1:
                    print(f"[Iter {it}/{epochs}] Loss: {loss.item():.4f}, NDCG: {ndcg:.4f}, HR: {hr:.4f}")

                if ndcg > best_ndcg:
                    best_ndcg, best_hr = ndcg, hr

                if active_edge_index is not None:
                    _, active_ndcg, _ = evaluation(self, active_edge_index, active_edge_index, [train_edge_index], K, LAMBDA)
                if inactive_edge_index is not None:
                    _, inactive_ndcg, _ = evaluation(self, inactive_edge_index, inactive_edge_index, [train_edge_index], K, LAMBDA)

            if it % per_lr_decay == 0 and it != 0:
                scheduler.step()

        elapsed_time = time.time() - start_time

        result = {
            "Recall": best_hr,
            "NDCG": best_ndcg,
            "HitRatio": best_hr,
            "active_ndcg": active_ndcg,
            "inactive_ndcg": inactive_ndcg,
            "time": elapsed_time
        }
        return self, result


# ======= Utils =======
def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val):
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) + pos_items_emb_0.norm(2).pow(2) + neg_items_emb_0.norm(2).pow(2))
    pos_scores = torch.sum(users_emb_final * pos_items_emb_final, dim=-1)
    neg_scores = torch.sum(users_emb_final * neg_items_emb_final, dim=-1)
    return -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss


def get_positive_items(edge_index):
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        user_pos_items.setdefault(user, []).append(item)
    return user_pos_items


def ndcgr(groundTruth, r, k):
    assert len(r) == len(groundTruth)
    test_matrix = torch.zeros((len(r), k))
    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    idcg = torch.sum(test_matrix / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = torch.sum(r / torch.log2(torch.arange(2, k + 2)), axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()


def hit(groundTruth, r, k):
    hits = [any(item in gt_items for item in r[i][:k]) for i, gt_items in enumerate(groundTruth)]
    return torch.mean(torch.tensor(hits).float()).item()


def get_metrics(model, edge_index, exclude_edge_indices, k):
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_positive_items(exclude_edge_index)
        for user, items in user_pos_items.items():
            rating[user, items] = -(1 << 10)

    _, top_K_items = torch.topk(rating, k=k)
    users = edge_index[0].unique()
    test_user_pos_items = get_positive_items(edge_index)
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]
    r = torch.Tensor(np.array([[item in test_user_pos_items[user.item()] for item in top_K_items[user]] for user in users]).astype('float'))

    ndcg = ndcgr(test_user_pos_items_list, r, k)
    hr = hit(test_user_pos_items_list, r, k)
    return ndcg, hr

def sample_batch(batch_size, edge_index):
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    
    indices = random.choices(range(edges.size(1)), k=batch_size)
    batch = edges[:, indices]
    
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices

def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(sparse_edge_index)
    edges = structured_negative_sampling(edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val)
    ndcg, hr = get_metrics(model, edge_index, exclude_edge_indices, k)
    return loss, ndcg, hr