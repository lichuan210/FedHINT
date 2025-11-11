import torch
import random

from lib.models.agcrn_HINT import  AGCRN_HINT


class Server():
    def __init__(self, args):

        self.model = AGCRN_HINT(num_nodes=args.num_nodes, input_dim=args.input_dim, rnn_units=args.rnn_units, output_dim=args.output_dim, history_seq_len=args.history_seq_len,
                                     horizon= args.future_seq_len, num_layers=args.num_layers, default_graph=True, embed_dim=args.embed_dim, cheb_k=args.cheb_k, att_dim=args.att_dim,num_k=args.knode).cuda()

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")

    def init_clients(self,clients):
        for client in clients:
            for k in self.W:
                if self.W[k].shape == client.W[k].shape:
                    client.W[k].data = self.W[k].data.clone()

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))


    def aggregate_weights_HINT(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.num_nodes
        #aggragation
        for k in self.W.keys():
            if 'global' in k:
                self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.num_nodes) for client in selected_clients]), dim=0), total_size).clone()
        #download
        for client in selected_clients:
            for k in self.W:
                if 'global' in k:
                    client.W[k].data = self.W[k].data.clone()

