from torch import nn

class GMF(nn.Module):
    def __init__(self, n_user, n_item, k=16):
        super(GMF, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)

        self.affine = nn.Linear(self.k, 1)
        self.logistic = nn.Sigmoid()

        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=STD)
        nn.init.normal_(self.item_mat.weight, std=STD)

        nn.init.xavier_uniform_(self.affine.weight)

    def forward(self, uid, iid):
        user_embedding = self.user_mat(uid)
        item_embedding = self.item_mat(iid)
        logits = self.affine(torch.mul(user_embedding, item_embedding))
        rating = self.logistic(logits)
        return rating.squeeze()
