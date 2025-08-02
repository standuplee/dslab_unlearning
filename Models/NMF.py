from torch import nn

class NMF(nn.Module):
    def __init__(self, n_user, n_item, k=16, layser=[64, 32]):
        super(NMF, self).__init__()
        self.k = k
        self.k_mlp = int(layser[0] / 2)

        self.user_mat_mf = nn.Embedding(n_user, k)
        self.item_mat_mf = nn.Embedding(n_item, k)
        self.user_mat_mlp = nn.Embedding(n_user, self.k_mlp)
        self.item_mat_mlp = nn.Embedding(n_item, self.k_mlp)

        self.layers = layser
        self.fc = nn.ModuleList()
        for (in_size, out_size) in zip(self.layers[:-1], self.layers[1:]):
            self.fc.append(nn.Linear(in_size, out_size))
            self.fc.append(nn.ReLU())

        self.affine = nn.Linear(self.layers[-1] + self.k, 1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat_mf.weight, std=STD)
        nn.init.normal_(self.item_mat_mf.weight, std=STD)
        nn.init.normal_(self.user_mat_mlp.weight, std=STD)
        nn.init.normal_(self.item_mat_mlp.weight, std=STD)

        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_uniform_(i.weight)
                if i.bias is not None:
                    i.bias.data.zero_()
        # for i in self.fc:
        #     if isinstance(i, nn.Linear):
        #         nn.init.xavier_uniform_(i.weight)
        #         if i.bias is not None:
        #             i.bias.data.zero_()

        # nn.init.xavier_uniform_(self.affine.weight)
        # if self.affine.bias is not None:
        #     self.affine.bias.data.zero_()

    def forward(self, uid, iid):
        user_embedding_mlp = self.user_mat_mlp(uid)
        item_embedding_mlp = self.item_mat_mlp(iid)

        user_embedding_mf = self.user_mat_mf(uid)
        item_embedding_mf = self.item_mat_mf(iid)

        mlp_vec = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vec = torch.mul(user_embedding_mf, item_embedding_mf)

        for i in range(len(self.fc)):
            mlp_vec = self.fc[i](mlp_vec)

        vec = torch.cat([mlp_vec, mf_vec], dim=-1)
        logits = self.affine(vec)
        rating = self.logistic(logits)
        return rating.squeeze()

