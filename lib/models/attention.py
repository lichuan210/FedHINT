from math import sqrt

import torch
import torch.nn as nn


class FullAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        #Q   B N_global H E
        #V&K B N_local H E
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayerV9(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer  like FilterNet
    '''

    def __init__(self, d_queries, d_keys, d_values, dim=64, n_heads=1, dropout=0.1):
        super(AttentionLayerV9, self).__init__()

        self.dim = dim
        self.inner_attention = FullAttention(scale=None, attention_dropout=dropout)

        self.query_projection = nn.Linear(d_queries, dim * n_heads)

        self.key_projection = nn.Linear(d_keys , dim * n_heads)
        self.key_weight = nn.Parameter(0.02 * torch.randn(288, dim * n_heads))

        self.value_projection = nn.Linear(d_values , dim * n_heads)
        self.value_weight = nn.Parameter(0.02 * torch.randn(288, dim * n_heads))

        self.n_heads = n_heads

    def forward(self, queries, keys, values,tod_index):
        queries = queries.unsqueeze(0).unsqueeze(0)  # 或 x[None, None, :, :]
        tod_index = tod_index.permute(0,2,3,1)
        queries = queries.expand(keys.shape[0], -1, -1, -1)
        queries = queries.permute(0,2,1,3)
        keys = keys.permute(0, 2, 3, 1)
        values = values.permute(0, 2, 3, 1)

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        E = keys.shape[-1]
        #DFT
        k_f = torch.fft.rfft(keys, dim=-1)
        v_f = torch.fft.rfft(values, dim=-1)

        index = tod_index[:,:,:,0]

        k_f =  k_f * torch.fft.rfft(self.key_weight[index.long(),:],dim=-1)
        v_f = v_f * torch.fft.rfft(self.value_weight[index.long(),:],dim=-1)

        keys = torch.fft.irfft(k_f, dim=-1,n=E)
        values = torch.fft.irfft(v_f, dim=-1,n=E)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        return out
