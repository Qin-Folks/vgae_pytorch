import torch
import torch.nn.functional as F
from dgl.data import TUDataset
from torch.nn import MSELoss
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import time
import scipy

from torch.optim.sgd import SGD

from input_data import load_data
from n_gram import get_gram_graph_embedding
from preprocessing import *
import args
import model
import sys

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""


def column_normalise(x):
    x = x.float()
    means = x.mean(0, keepdim=True)
    stdevs = x.std(0, keepdim=True)
    stdevs[stdevs == 0] = 1

    x_normed = (x - means) / stdevs
    return x_normed


def adjust_learning_rate(optim):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optim.param_groups:
        param_group['lr'] = param_group['lr'] / 10


dataset = TUDataset(name='PROTEINS_full')
a_molecule = dataset[0][0]
normed_node_features = column_normalise(a_molecule.ndata['node_attr'])[:, :10]
# print('a molecule nodes: ', normed_node_features)
# print('a molecule adj: ', a_molecule.adjacency_matrix())
adj = scipy.sparse.csr.csr_matrix(a_molecule.adjacency_matrix().to_dense().numpy())
features = scipy.sparse.lil_matrix(normed_node_features.numpy())
# print('type of features: ', type(features))
# print('type of adj: ', type(adj))

# adj, features = load_data(args.dataset)
# print('type of features: ', type(features))
# print('type of adj: ', type(adj))

label_feature = None
# if args.dataset.lower() == 'cora':
#     label_feature = features[:, -1]
#     features = features[:, :-1]

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create Model
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                    torch.FloatTensor(adj_norm[1]),
                                    torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                     torch.FloatTensor(adj_label[1]),
                                     torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
                                    torch.FloatTensor(features[1]),
                                    torch.Size(features[2]))

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight

# init model and optimizer
model = getattr(model, args.model)(adj_norm)
# optimizer = Adam(model.parameters(), lr=args.learning_rate)
optimizer_adam = Adam(model.parameters(), lr=1e-3)
optimizer_sgd = SGD(model.parameters(), lr=1e-5, momentum=0.9)


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        # print(e)
        # print(adj_rec[e[0], e[1]])
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


# train model
optimizer = optimizer_adam
for epoch in range(args.num_epoch):
    if epoch == 1000:
        optimizer = optimizer_sgd
    if epoch > 1000 and epoch % 100 == 0:
        adjust_learning_rate(optimizer)
    t = time.time()

    A_pred = model(features)
    x_ones = torch.ones_like(features.to_dense())
    # print('A pred: \n', (A_pred > 0.5).int())
    # print('adj label: \n', adj_label.to_dense())
    pred_ngram = get_gram_graph_embedding(x_ones, A_pred, is_soft=True)
    orig_ngram = get_gram_graph_embedding(x_ones, adj_label.to_dense(), is_soft=False)

    optimizer.zero_grad()
    # loss = log_lik = norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
    loss = MSELoss()(pred_ngram, orig_ngram)
    # if args.model == 'VGAE':
    #     kl_divergence = 0.5 / A_pred.size(0) * (1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd)).sum(
    #         1).mean()
    #     loss -= kl_divergence

    loss.backward()
    optimizer.step()

    train_acc = get_acc(A_pred, adj_label)

    val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
          "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
          "val_ap=", "{:.5f}".format(val_ap),
          "time=", "{:.5f}".format(time.time() - t))

test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
      "test_ap=", "{:.5f}".format(test_ap))
