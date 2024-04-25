import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
class GraphRec(nn.Module):

    def __init__(self, User_factor, Item_factor, embed_dim):
        super(GraphRec, self).__init__()
        self.embed_dim = embed_dim
        self.User_factor = User_factor
        self.Item_factor = Item_factor
        self.ln_h1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.bn_h1 = nn.BatchNorm1d(self.embed_dim , momentum=0.5)
        self.ln_h2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.ln_z1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.bn_z1 = nn.BatchNorm1d(self.embed_dim , momentum=0.5)
        self.ln_z2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.ln_g1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.bn_g1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.ln_g2 = nn.Linear(self.embed_dim, 32)
        self.bn_g2 = nn.BatchNorm1d(32, momentum=0.5)
        self.ln_g3 = nn.Linear(32, 16)
        self.bn_g3 = nn.BatchNorm1d(16, momentum=0.5)
        self.ln_g4 = nn.Linear(16, 1)

    def forward(self, social_adj, item_history, itemrating_history, user_nodes,
                item, user_history, userrating_history, item_nodes):
        h_i = self.User_factor.social_final(social_adj, item_history, itemrating_history, user_nodes)
        z_i = self.Item_factor.user_final(item, user_history, userrating_history, item_nodes)

        h_i = F.dropout(F.relu(self.bn_h1(self.ln_h1(h_i))), training=self.training)
        h_i = self.ln_h2(h_i)
        z_i = F.dropout(F.relu(self.bn_z1(self.ln_z1(z_i))), training=self.training)
        z_i = self.ln_z2(z_i)

        g = torch.cat([h_i,z_i],1)
        g = F.dropout(F.relu(self.bn_g1(self.ln_g1(g))), training=self.training)
        g = F.dropout(F.relu(self.bn_g2(self.ln_g2(g))), training=self.training)
        g = F.dropout(F.relu(self.bn_g3(self.ln_g3(g))), training=self.training)
        rating = self.ln_g4(g)
        return rating.squeeze()

    def loss(self, social_adj, item_history, itemrating_history, user_nodes,
             item, user_history, userrating_history, item_nodes, label_rating):
        prediction = self.forward(social_adj, item_history, itemrating_history, user_nodes,
                                  item, user_history, userrating_history, item_nodes)
        return nn.MSELoss()(prediction, label_rating)