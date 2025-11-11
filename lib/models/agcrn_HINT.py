import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import *

class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, mask):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.mask = mask.cuda()
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(
                2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)

        supports = supports * self.mask
        # N, cheb_k, dim_in, dim_out
        weights = torch.einsum(
            'nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports,
                           x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g,
                               weights) + bias  # b, N, dim_out
        return x_gconv

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, mask):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2 *
                           dim_out, cheb_k, embed_dim, mask)
        self.update = AVWGCN(dim_in+self.hidden_dim,
                             dim_out, cheb_k, embed_dim, mask)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size,N):
        return torch.zeros(batch_size, N, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1, mask=None):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(
            AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim, mask))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim, mask))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, shape):
        batch_size = shape[0]
        N = shape[2]
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.dcrnn_cells[i].init_hidden_state(batch_size,N))
        # (num_layers, B, N, hidden_dim)
        return torch.stack(init_states, dim=0)


class AGCRN_HINT(nn.Module):
    """
    Paper: Adaptive Graph Convolutional Recurrent Network for Trafﬁc Forecasting
    Official Code: XXXX
    Link: XXXX
    """

    def __init__(self, num_nodes, input_dim, rnn_units, output_dim, history_seq_len ,horizon, num_layers, default_graph, embed_dim, cheb_k,att_dim,num_k):
        super(AGCRN_HINT, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.default_graph = default_graph
        self.num_global_nodes = num_k
        self.att_dim = att_dim

        self.node_embeddings = nn.Parameter(torch.randn(
            self.num_node, embed_dim), requires_grad=True)

        self.global_node_embeddings = nn.Parameter(torch.randn(
            self.num_global_nodes, embed_dim), requires_grad=True)

        self.local_mask = (torch.arange(self.num_node+self.num_global_nodes).view(self.num_node+self.num_global_nodes, 1) < self.num_global_nodes) & (torch.arange(self.num_node+self.num_global_nodes) < self.num_global_nodes)

        self.local_encoder = AVWDCRNN(num_nodes, input_dim, rnn_units, cheb_k,
                                embed_dim, num_layers, mask = self.local_mask)

        #### Global Parameters ####
        self.global_attention = AttentionLayerV9(d_queries=embed_dim, d_keys=history_seq_len, d_values=history_seq_len, dim=att_dim)
        self.global_project = nn.Linear(att_dim, history_seq_len)
        self.global_encoder = AVWDCRNN(num_nodes, input_dim, rnn_units, cheb_k,
                                       embed_dim, num_layers, mask = ~self.local_mask)

        # predictor
        self.end_conv = nn.Conv2d(
            1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim*2), bias=True)

        self.init_param()

    def init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, history_data):
        """Feedforward function of AGCRN.

        Args:
            history_data (torch.Tensor): inputs with shape [B, L, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        tod_index = history_data[:,:,:,1].unsqueeze(-1)
        history_data = history_data[:,:,:,0].unsqueeze(-1)
        global_feature = self.global_attention(self.global_node_embeddings, history_data, history_data,tod_index)
        global_feature = self.global_project(global_feature).permute(0,3,1,2)
        full_data = torch.concat((history_data, global_feature), dim=2)
        full_node_embeddings = torch.concat((self.node_embeddings, self.global_node_embeddings), dim=0)

        #local
        local_init_state = self.local_encoder.init_hidden(full_data.shape)
        local_output, _ = self.local_encoder(
            full_data, local_init_state, full_node_embeddings)  # B, T, N, hidden
        local_output = local_output[:, -1:, :, :]  # B, 1, N, hidden

        #global
        global_init_state = self.global_encoder.init_hidden(full_data.shape)
        global_output, _ = self.global_encoder(
            full_data, global_init_state, full_node_embeddings)  # B, T, N, hidden
        global_output = global_output[:, -1:, :, :]  # B, 1, N, hidden

        output = torch.concat((local_output, global_output), dim=-1)
        output = output[:,:,0:self.num_node,:]
        # Linear predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon,
                                            self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output, self.global_node_embeddings
