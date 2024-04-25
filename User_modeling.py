import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import Attention

class Aggre_item(nn.Module):
    def __init__(self, user, item, rating, embed_dim, cuda):
        super(Aggre_item,self).__init__()
        self.user = user
        self.item = item
        self.rating = rating
        self.embed_dim = embed_dim
        self.device = cuda
        self.ln1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.ln2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.ln3 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.item_att = Attention(self.embed_dim)

    # user model: item aggregation
    def item_aggregation(self, item_history, itemrating_history, nodes):
        '''
        @param.nodes: the nodes' list of users
        @param.history: lists of item and rating
        len(item_history) == len(nodes)
        '''
        embed_matrix = torch.empty((len(item_history), self.embed_dim), dtype=torch.float , device=self.device)
        for i in range(len(item_history)):
            item_list = item_history[i]
            num_item = len(item_list)
            rating_list = itemrating_history[i]

            w_item = self.item.weight[item_list]
            w_rating = self.rating.weight[rating_list]
            w_user = self.user.weight[nodes[i]]

            #x_ia = [q_a(+)e_r]
            x_i = torch.cat([w_item,w_rating],1)
            x_i = F.relu(self.ln1(x_i))   # shape of x_i: num_item * embed_dim

            #alpha = att{ W2 * σ(W1[x_ia(+)p_i] + b1) + b2 ) }
            w_user = w_user.repeat(num_item, 1)
            alpha = torch.cat([x_i,w_user], 1)
            alpha = self.item_att(alpha)   # shape of alpha: num_item * 1

            #hI_i = σ{ W * (x_i.t * alpha) + b }
            hI_i = torch.mm(x_i.t(), alpha)
            hI_i = F.relu(self.ln2(hI_i.t()))   # shape of hI_i: 1 * embed_dim

            embed_matrix[i] = hI_i

        return embed_matrix

    #user model: Item space final
    def item_final(self, user, nodes, item_history, itemrating_history):
        # @para.nodes: the nodes' list of users

        item_history_temp = []
        itemrating_history_temp = []
        for node in nodes:
            item_history_temp.append(item_history[node])
            itemrating_history_temp.append(itemrating_history[node])

        item_feature = self.item_aggregation(item_history_temp, itemrating_history_temp, nodes)
        user_feature = user.weight[nodes]

        #combine
        features = torch.cat([user_feature, item_feature], 1)
        features = F.relu(self.ln3(features))
        return features

class Aggre_social(nn.Module):
    def __init__(self, user, item, rating, embed_dim, cuda):
        super(Aggre_social,self).__init__()
        self.user = user
        self.item = item
        self.rating = rating
        self.embed_dim = embed_dim
        self.device = cuda
        self.ln1 = nn.Linear(embed_dim, embed_dim)
        self.ln2 = nn.Linear(embed_dim * 2, embed_dim)
        self.ln3 = nn.Linear(embed_dim, embed_dim)
        self.social_att = Attention(self.embed_dim)
        self.item_agg = Aggre_item(self.user , self.item , self.rating , self.embed_dim, self.device)

    #user model: social aggregation
    def social_Aggregation(self, social_history, item_history, itemrating_history, nodes):
        #@param.nodes: the nodes' list of users
        #@param.social_history: lists of whom connected with the users

        embed_matrix = torch.empty((len(social_history), self.embed_dim), dtype=torch.float, device=self.device)
        for i in range(len(social_history)):
            social_list = social_history[i]
            num_social_user = len(social_list)

            #hI_i and p_i
            hI_i = self.item_agg.item_final(self.item_agg.user, social_list, item_history, itemrating_history)
            w_user = self.user.weight[nodes[i]]
            ''' shape of hI_i: len(item_history_temp) * embed_dim
                              =num_social_user * embed_dim
                shape of w_user: 1 * embed_dim '''
            # beta = att{ W2 * σ(W1[hI_i(+)p_i] + b1) + b2 ) }
            w_user = w_user.repeat(num_social_user, 1)
            beta = torch.cat([hI_i,w_user],1)
            beta = self.social_att(beta)
            #hS_i = σ{ W * (hI_i * beta) + b }
            hS_i = torch.mm(hI_i.t(),beta)
            hS_i = F.relu(self.ln1(hS_i.t()))

            embed_matrix[i] = hS_i

        return embed_matrix

    #User model final
    def social_final(self, social_history, item_history, itemrating_history, nodes):
        # @para.nodes: the nodes' list of users

        social_history_temp = []

        for node in nodes.tolist():
            social_history_temp.append(social_history[node])
        #print(type(nodes))
        hI_i = self.item_agg.item_final(self.item_agg.user, nodes, item_history, itemrating_history)
        hS_i = self.social_Aggregation(social_history_temp, item_history, itemrating_history, nodes)
        #features = hI_i (+) hS_i
        User_factors = torch.cat([hI_i, hS_i], 1)
        User_factors = F.relu(self.ln2(User_factors))
        User_factors = F.relu(self.ln3(User_factors))

        return User_factors