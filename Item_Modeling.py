import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import Attention

class Aggre_user(nn.Module):
    def __init__(self, user, item, rating, embed_dim, cuda):
        super(Aggre_user,self).__init__()
        self.user = user
        self.item = item
        self.rating = rating
        self.embed_dim = embed_dim
        self.device = cuda
        self.ln1 = nn.Linear(embed_dim * 2, embed_dim)
        self.ln2 = nn.Linear(embed_dim, embed_dim)
        self.ln3 = nn.Linear(embed_dim * 2, embed_dim)
        self.user_att = Attention(embed_dim)

    def user_Aggregation(self, user_history, userrating_history, nodes):

        embed_matrix = torch.empty((len(user_history), self.embed_dim), dtype=torch.float, device=self.device)
        for j in range(len(user_history)):
            user_list = user_history[j]
            num_user = len(user_list)
            userrating_list = userrating_history[j]

            w_user = self.user.weight[user_list]
            w_rating = self.rating.weight[userrating_list]      # num_user * embed_dim
            w_item = self.item.weight[nodes[j]]     # 1 * embed_dim

            #f_j = [p_t (+) e_r]
            f_j = torch.cat([w_user, w_rating], 1)
            f_j = F.relu(self.ln1(f_j))     # shape of f_t: num_user * embed_dim

            #mu = att{ W2 * σ(W1 * [f_jt (+) q_j] + b1) +b2 }
            w_item = w_item.repeat(num_user, 1)
            mu = torch.cat([f_j, w_item], 1)
            mu = self.user_att(mu)      # shape of mu: num_user * 1

            #z_j = σ{ W * (f_j.t * mu) + b }
            z_j = torch.mm(f_j.t(), mu)
            z_j = F.relu(self.ln2(z_j.t()))     # shape of z_j: 1 * embed_dim

            embed_matrix[j] = z_j
        return embed_matrix

    #Item model final
    def user_final(self, item, user_history, userrating_history, nodes):
        # @para.nodes: the nodes' list of items

        user_history_temp = []
        userrating_history_temp = []

        for node in nodes:
            user_history_temp.append(user_history[node])
            userrating_history_temp.append(userrating_history[node])

        user_feature = self.user_Aggregation(user_history_temp, userrating_history_temp, nodes)
        item_feature = item.weight[nodes]

        #combine
        Item_factors = torch.cat([item_feature, user_feature],1)
        Item_factors = F.relu(self.ln3(Item_factors))

        return Item_factors