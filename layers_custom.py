import torch
import torch.nn.functional as F
import torch.nn as nn

class AKConv(torch.nn.Module):
    def __init__(self):
        super(AKConv, self).__init__()
        self.lambda_ = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.special_spmm = SpecialSpmm()
        self.i = None
        self.v_1 = None
        self.v_2 = None
        self.nodes = None

    def get_actual_lambda(self):
        return 1 + torch.relu(self.lambda_)
    
    def forward(self, input, adj):
        lambda_ = self.get_actual_lambda()

        if self.i == None:
        	self.nodes = adj.shape[0]
        	dummy = [i for i in range(self.nodes)]
        	i_1 = torch.tensor([dummy, dummy]).cuda()
        	i_2 = adj.coalesce().indices().cuda()
        	self.i = torch.cat([i_1, i_2], dim = 1)
        	self.v_1 = torch.tensor([1 for _ in range(self.nodes)]).cuda()
        	self.v_2 = torch.tensor([1 for _ in range(len(i_2[0]))]).cuda()

        v_1 = ((2 * lambda_ - 2) / lambda_) * self.v_1
        v_2 = (2 / lambda_) * self.v_2
        v = torch.cat([v_1, v_2])
        e_rowsum = self.special_spmm(self.i, v, torch.Size([self.nodes, self.nodes]), torch.ones(size=(self.nodes,1)).cuda())
        return self.special_spmm(self.i, v, torch.Size([self.nodes, self.nodes]), input).div(e_rowsum)

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
