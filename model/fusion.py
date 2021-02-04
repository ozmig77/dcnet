import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def get_fusion(fusion_type, *args, **kwargs):
    Fusion = {
        'hadamard': Hadamard,
        'concat': Concat,
        'mutan': Mutan,
        'mlb': MLB,
        'mcb': MCB,
    }[fusion_type.lower()]
    return Fusion(*args, **kwargs)

class Concat(nn.Module):
    def __init__(self, emb_dims, *args, **kwargs):
        super(Concat, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dims*2, emb_dims),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        assert x1.size(1) == x2.size(1)
        x = torch.cat([x1, x2], -1)
        x = self.fc(x)
        return x

class Hadamard(nn.Module):
    def __init__(self):
        super(Hadamard, self).__init__()

    def forward(self, x1, x2):
        assert x1.size(1) == x2.size(1)
        x = x1 * x2
        return x


class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dims, output_dim, sum_pool=True):
        super(CompactBilinearPooling, self).__init__()

        input_dim1 = input_dims
        input_dim2 = input_dims
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        generate_sketch_matrix = lambda rand_h, rand_s, input_dim, output_dim: \
            torch.sparse.FloatTensor(torch.stack(
                [torch.arange(input_dim, out = torch.LongTensor()), rand_h.long()]),
                rand_s.float(), [input_dim, output_dim]).to_dense()
        self.sketch1 = torch.nn.Parameter(generate_sketch_matrix(
            torch.randint(output_dim, size = (input_dim1,)),
            2 * torch.randint(2, size = (input_dim1,)) - 1, input_dim1, output_dim),
            requires_grad = False)
        self.sketch2 = torch.nn.Parameter(generate_sketch_matrix(
            torch.randint(output_dim, size = (input_dim2,)),
            2 * torch.randint(2, size = (input_dim2,)) - 1, input_dim2, output_dim),
            requires_grad = False)

    def forward(self, x1, x2):
        fft1 = torch.rfft(x1.matmul(self.sketch1), signal_ndim = 1)
        fft2 = torch.rfft(x2.matmul(self.sketch2), signal_ndim = 1)
        fft_product = torch.stack(
            [fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1],
             fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
        cbp = torch.irfft(fft_product, signal_ndim = 1,
                          signal_sizes = (self.output_dim, )) * self.output_dim
        return cbp.sum(dim=[1, 2]) if self.sum_pool else cbp


class Mutan(nn.Module):
    def __init__(self, m_dim, input_dims, out_dim, chunks, head, dropout):
        super(Mutan, self).__init__()

        self.head = head # 12
        self.normalize = True
        self.dropout = dropout

        self.reduce_q = nn.Linear(input_dims, m_dim, bias=False)
        self.reduce_v = nn.Linear(input_dims, m_dim, bias=False)

        self.linear_q = nn.Linear(m_dim, m_dim * self.head)
        self.linear_v = nn.Linear(m_dim, m_dim * self.head)

        self.out = nn.Linear(m_dim, out_dim)
    '''
    def out(self, z):
        return torch.matmul(z, self.reduce_v.weight)
    '''

    def forward(self, q, v):
        if len(q.shape) != len(v.shape):
            q = q.unsqueeze(1)
        q = self.reduce_q(q)
        v = self.reduce_v(v)

        if self.dropout > 0:
            q = F.dropout(q, p=self.dropout)
            v = F.dropout(v, p=self.dropout)

        q = self.linear_q(q)
        v = self.linear_v(v)

        q = rearrange(q, 'b ... (h c) -> b ... h c', h=self.head)
        v = rearrange(v, 'b ... (h c) -> b ... h c', h=self.head)
        m = q * v
        m = torch.sum(m, -2)

        if self.normalize:
            m = torch.sqrt(F.relu(m)) - torch.sqrt(F.relu(-m))
            m = F.normalize(m, p=2)
        z = self.out(m)

        return z


# recommeded m_dim: 1200
class MLB(nn.Module):
    def __init__(self, m_dim, input_dims, out_dim, chunks, head, dropout):
        super(MLB, self).__init__()

        self.head = head
        self.normalize = True
        self.dropout = dropout

        self.reduce_q = nn.Linear(input_dims, m_dim, bias=False)
        self.reduce_v = nn.Linear(input_dims, m_dim, bias=False)

        self.out = nn.Linear(m_dim, out_dim)

    def forward(self, q, v):
        if len(q.shape) != len(v.shape):
            q = q.unsqueeze(1)
        q = self.reduce_q(q)
        v = self.reduce_v(v)

        if self.dropout > 0:
            q = F.dropout(q, p=self.dropout)
            v = F.dropout(v, p=self.dropout)

        m = q * v

        if self.normalize:
            m = torch.sqrt(F.relu(m)) - torch.sqrt(F.relu(-m))
            m = F.normalize(m, p=2)
        z = self.out(m)

        return z


# recommended m_dim: 16000
class MCB(nn.Module):
    def __init__(self, m_dim, input_dims, out_dim, chunks, head, dropout):
        super(MCB, self).__init__()

        self.head = head
        self.normalize = True
        self.dropout = dropout

        self.mcb = CompactBilinearPooling(input_dims, m_dim, sum_pool=False)
        self.out = nn.Linear(m_dim, out_dim)

    def forward(self, q, v):
        if len(q.shape) != len(v.shape):
            q = q.unsqueeze(1)

        m = self.mcb(q, v)

        if self.normalize:
            m = torch.sqrt(F.relu(m)) - torch.sqrt(F.relu(-m))
            m = F.normalize(m, p=2)

        z = self.out(m)
        if self.dropout > 0:
            z = F.dropout(z, p=self.dropout)

        return z