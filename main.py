import pickle
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from User_modeling import Aggre_social
from Item_Modeling import Aggre_user
from Model import GraphRec

def train(model, optimizer, trainloader, social_adj, item_history, itemrating_history,
          item, user_history, userrating_history, epoch, device):
    model.train()
    train_loss = 0.0
    start_time = time.time()
    for i, data in enumerate(trainloader, 0):
        user_nodes, item_nodes, label_rating = data
        user_nodes, item_nodes, label_rating = user_nodes.to(device), item_nodes.to(device), label_rating.to(device)
        optimizer.zero_grad()
        loss = model.loss(social_adj, item_history, itemrating_history, user_nodes,
                          item, user_history, userrating_history, item_nodes, label_rating)     #i.e: do combine once
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
        if i % 100 == 0 and i != 0:
            print('{epoch:%d, trained_batch:%d}, {train_loss: %.3f, each_batch_time:%.3fs}'
                  % (epoch, i, train_loss/i, (time.time() - start_time)/i))
            train_loss = 0.0
            start_time = time.time()
    return 0


def test(model, testloader, social_adj, item_history, itemrating_history,
         item, user_history, userrating_history, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            user_nodes, item_nodes, label_rating = data
            user_nodes, item_nodes, label_rating = user_nodes.to(device), item_nodes.to(device), label_rating.to(device)
            error = model.forward(social_adj, item_history, itemrating_history, user_nodes,
                                  item, user_history, userrating_history, item_nodes)
            error = torch.abs(error - label_rating)
            errors.extend(error.cpu().tolist())
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.power(errors, 2)))
    return mae, rmse

def main():
    #setting parameters
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int,  default=1000, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embed_dim = args.embed_dim

    #import data
    data = open('dataset.pickle', 'rb')
    (user_history, userrating_history, item_history, itemrating_history,
     train_user, train_item, train_rating, test_user, test_item, test_rating,
     social_adj0, rating_list) = pickle.load(data)

    social_adj_temp = {}
    for node, data in social_adj0.items():
        social_adj_temp[node] = list(data)
    social_adj = social_adj_temp

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_user), torch.LongTensor(train_item),
                                torch.FloatTensor(train_rating))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_user), torch.LongTensor(test_item),
                               torch.FloatTensor(test_rating))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    user = nn.Embedding(len(user_history), embed_dim).to(device)
    item = nn.Embedding(len(item_history), embed_dim).to(device)
    rating = nn.Embedding(len(rating_list), embed_dim).to(device)

    User_factor = Aggre_social(user, item, rating, embed_dim, device)
    Item_factor = Aggre_user(user, item, rating, embed_dim, device)

    model = GraphRec(User_factor, Item_factor, embed_dim).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)

    best_mae = 10000.0
    count = 0
    for epoch in range(1, args.epochs):
        train(model, optimizer, trainloader, social_adj, item_history, itemrating_history,
              item, user_history, userrating_history, epoch, device)
        mae, rmse = test(model, testloader, social_adj, item_history, itemrating_history,
                         item, user_history, userrating_history, device)
        if mae < best_mae:
            best_mae = mae
            count = 0
        else:
            count += 1
        print('{epoch:%d, rmse=%.3f}, {best_mae=%.3f}' % (epoch, rmse, best_mae))
        if count >= 5:
            break
    return 0


if __name__ == '__main__':
    main()
