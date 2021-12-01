# This code is copyied and modified from PyGCN the official GitHub of GCN
# https://github.com/tkipf/pygcn


from __future__ import division
from __future__ import print_function

import time
import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_pygcn import accuracy, sparse_mx_to_torch_sparse_tensor
from utils import load_data

from models import AKGNN
import numpy as np

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=31, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--device', type=int, default=0,
                    help='which GPU to run')
parser.add_argument('--early_patience', type=int, default=100)
parser.add_argument('--layers', type=int, default=5)


args = parser.parse_args()
args.cuda = True #not args.no_cuda and torch.cuda.is_available()

torch.cuda.set_device(args.device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Load data pygcn version
adj, features, labels, idx_train, idx_val, idx_test = load_data('citeseer', args.seed, 20)

# Model and optimizer
model = AKGNN(
        n_layer = args.layers,
        in_dim=features.shape[1],
        h_dim=args.hidden,
        n_class=labels.max().item() + 1,
        activation = F.leaky_relu,
        dropout=args.dropout
        )
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results under current model:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test


# initialize placeholders for early stopping
best_val = torch.tensor(100.)
best = None
best_epoch = 0
best_test = 0
# Train model
t_total = time.time()
for epoch in range(args.epochs):
    if epoch - best_epoch >= args.early_patience:
        break
    loss_val = train(epoch)
    if loss_val < best_val:
        best_epoch = epoch
        try:
            print(epoch, 'A BETTER VALIDATION FOUND:', best_val.detach().cpu().numpy(), '->', loss_val.detach().cpu().numpy())
            best_test = test()
            best = [float(layer.get_actual_lambda().detach().cpu().numpy()) for layer in model.layers]
        except:
            continue
        best_val = loss_val

print('Test ACC:', float(best_test.detach().cpu().numpy()))
print('lambda max at each layer:')
print(best)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))