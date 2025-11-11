import time
import numpy as np
import torch.multiprocessing as mp


def run_fedHINT(logger , clients, server, rounds, local_epoch, samp=None, frac=1.0):
    # all clients are initialized with the same weights
    server.init_clients(clients)

    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0

    for round in range(1, rounds + 1):
        logger.info(f"  > round {round}")
        if round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)

        for client in selected_clients:
            client.local_train_HINT(local_epoch)

        server.aggregate_weights_HINT(selected_clients)
