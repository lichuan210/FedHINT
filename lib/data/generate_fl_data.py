import pymetis
import networkx as nx
import numpy as np

from lib.data.load_dataset import load_st_dataset
from lib.data.dataloader import get_dataloader
from datasets.dividing import *

def get_fl_dataset(args):
    data, adj_matrix = load_st_dataset(args.dataset)
    args.logger.info(f'Load {args.dataset} Dataset shaped: {data.shape}')

    # graph partition
    try:
        partition = eval(f"{args.dataset}_{args.num_clients}p_{args.divide}")
    except:
        if args.num_clients == adj_matrix.shape[0] :
            partition = [[i] for i in range(args.num_clients)]
            with open("./datasets/dividing.py", "a") as f:
                f.write(f"\n{args.dataset}_{args.num_clients}p_{args.divide} = [[i] for i in range({args.num_nodes})]\n")
        elif args.num_clients == 1:
            partition = [[i for i in range(args.num_nodes)]]
            with open("./datasets/dividing.py", "a") as f:
                f.write(
                    f"\n{args.dataset}_{args.num_clients}p_{args.divide} = [[i for i in range({args.num_nodes})]]\n")
        else:
            G = nx.from_numpy_array(adj_matrix)
            n_cuts, part = pymetis.part_graph(args.num_clients, G)
            partition = [list(np.where(np.array(part) == i)[0]) for i in range(args.num_clients)]
            with open("./datasets/dividing.py", "a") as f:
                f.write(f"\n{args.dataset}_{args.num_clients}p_{args.divide} = {list(partition)}\n")

    clients_data = []
    for client_id in range(args.num_clients):
        nodes = partition[client_id]
        client_adj = adj_matrix[nodes][:, nodes]
        args.logger.info(f'Loading Data for Client: {client_id}' )
        train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(data[:, nodes], args)
        clients_data.append({
            "adj":client_adj,
            "train_loader": train_dataloader,
            "val_loader": val_dataloader,
            "test_loader": test_dataloader,
            "scaler":scaler
        })

    return clients_data