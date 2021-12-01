import torch.nn as nn
import torch.nn.functional as F
from layers_custom import AKConv
import torch
import math


class AKGNN(torch.nn.Module):
    def __init__(self, n_layer, in_dim, h_dim, n_class, activation, dropout):
        super(AKGNN, self).__init__()
        # list of propagation layers
        self.layers = torch.nn.ModuleList([AKConv() for _ in range(n_layer)])
        # theta parameter which approximates theta0 * theta1 ... thetak
        self.theta = torch.nn.Linear(in_dim, h_dim)
        # MLP predictor
        self.predictor = torch.nn.Linear(h_dim * n_layer, n_class)
        self.activation = activation
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta.weight.size(1))
        self.theta.weight.data.uniform_(-stdv, stdv)
        self.theta.bias.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.predictor.weight.size(1))
        self.predictor.weight.data.uniform_(-stdv, stdv)
        self.predictor.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        h = self.activation(self.theta(input))
        h = F.dropout(h, self.dropout, training=self.training)
        h_list = []
        # iteratively append output of each layer to h_list
        for propagation_layer in self.layers:
            h = propagation_layer(h, adj)
            h = F.dropout(h, self.dropout, training=self.training)
            h_list.append(h)
        h = torch.cat(h_list, dim = 1)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.predictor(h)
        return F.log_softmax(h, dim=1)