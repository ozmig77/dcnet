import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=1, fix_norm=False):
        super().__init__()
        self.fix_norm = fix_norm
        #self.loss = torch.nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x):
        # x: B x B
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1) # n^2, 1
        #x1 = torch.cat((x1, x1), 0) # 2*n^2, 1

        x2 = x.view(-1, 1)
        #x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        #x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()


class TripletLoss(nn.Module):

    def __init__(self, margin=0.2, max_violation=False, cost_style='sum', direction='all'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        self.max_violation = max_violation

    def forward(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in  ['i2t', 'all']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'all']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]

        if cost_s is None:
            cost_s = Variable(torch.zeros(1)).cuda()
        if cost_im is None:
            cost_im = Variable(torch.zeros(1)).cuda()

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()


class AdditiveMarginSoftmax(nn.Module):
    def __init__(self, margin=0.3, dual=False):
        super(AdditiveMarginSoftmax, self).__init__()
        self.margin = margin
        self.dual = dual

    def forward(self, x):
        B, N = x.size()

        tmp_label = torch.arange(B).long().cuda()
        margin_mat = torch.diag(torch.ones(N) * self.margin).cuda()
        x = x - margin_mat[:B, :N]

        # forward
        f_loss = F.cross_entropy(x, tmp_label)

        if self.dual:
            # backward
            b_loss = F.cross_entropy(x[:B,:B].t(), tmp_label)
            return f_loss + b_loss
        else:
            return f_loss
