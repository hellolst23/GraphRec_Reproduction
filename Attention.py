import torch.nn.functional as F
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention,self).__init__()
        self.embed_dim = embed_dim
        self.ln1 = nn.Linear(embed_dim * 2, embed_dim)
        self.ln2 = nn.Linear(embed_dim, embed_dim)
        self.ln3 = nn.Linear(embed_dim, 1)


    def forward(self, alpha):
        alpha = F.relu(self.ln1(alpha))
        alpha = F.dropout(alpha, training=self.training)
        alpha = F.relu(self.ln2(alpha))
        alpha = F.dropout(alpha, training=self.training)
        alpha = self.ln3(alpha)
        alpha = F.softmax(alpha, 1)

        return alpha    # a len(alpha) * 1 vector
