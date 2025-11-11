import random
import torch
import argparse
import os
import sys
from federated import *
from lib.data.generate_fl_data import get_fl_dataset
from lib.logger import get_logger
from lib.client import Client
from lib.server import Server
import json



def set_cpu_num(cpu_num):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    server = Server(args)

    clients_list = []
    clients_data = get_fl_dataset(args)
    for idx, data in enumerate(clients_data):
        client = Client(data, idx, args)
        clients_list.append(client)

    run_fedHINT(args.logger, clients_list, server, rounds=args.rounds, local_epoch=args.epochs)

    #save last model
    for client in clients_list:
        torch.save(client.model.state_dict(), args.save_dir + '/Client' + str(client.client_id) + str(args.rounds) +'.pth')



CONFIG = sys.argv[1]
del sys.argv[1]


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='arguments')

    args.add_argument('--mode', type=str, default="FedHINT")
    args.add_argument('--num_clients', type=int, default=1, help="number of clients")
    args.add_argument('--save_dir', type=str, default="save")
    args.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    args.add_argument('--cpu_num', type=int, default=4, help='cpu num')
    args.add_argument('--seed', type=int, default=12, help='random seed')
    args.add_argument('--embed_dim', type=int, default=10)
    args.add_argument('--divide', type=str, default='metis', help="dividing")
    args.add_argument('--knode', type=int, default=32, help="num of knode")

    args = args.parse_args()

    #get configuration
    with open(CONFIG, 'r') as f:
        config = json.load(f)

    for section, dict in config.items():
        for key, value in dict.items():
            setattr(args, key, value)

    set_cpu_num(args.cpu_num)
    set_seed(args.seed)

    args.save_dir = f'./save/{args.save_dir}/{args.mode}/{args.dataset}/{str(args.num_clients)}'
    args.logger = get_logger(args=args)
    # print args
    args.logger.info("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        args.logger.info("|{0:>20} = {1:<45}|".format(key, str(value)))
    args.logger.info("-"*(20+45+5))

    main(args)
