from torch import nn

class BPR(nn.Module):
    def __init__(self, n_user, n_item, k=16):
        super(BPR, self).__init__()
        self.k = k
        self.user_mat = nn.Embedding(n_user, k)
        self.item_mat = nn.Embedding(n_item, k)
        self.func = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.user_mat.weight, std=0.01)
        nn.init.normal_(self.item_mat.weight, std=0.01)

    def forward(self, uid, iid):
        return (self.user_mat(uid) * self.item_mat(iid)).sum(dim=1)

        # r_pos = (self.user_mat(uid) * self.item_mat(pos_id)).sum(dim=1)
        # r_neg = (self.user_mat(uid) * self.item_mat(neg_id)).sum(dim=1)

        # return self.func(r_pos - r_neg)