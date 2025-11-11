from lib.untils import *
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.agcrn_HINT import  AGCRN_HINT



class Client():
    def __init__(self,client_data, client_id, args):
        self.logger = args.logger
        self.device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
        self.client_data = client_data
        self.client_id = client_id
        self.num_nodes = client_data["adj"].shape[1]
        self.loss = torch.nn.L1Loss().to(self.device)
        self.best_val_loss = float('inf')
        self.lr_decay = args.lr_decay
        self.args = args


        self.model = AGCRN_HINT(num_nodes=self.num_nodes, input_dim=args.input_dim, rnn_units=args.rnn_units, output_dim=args.output_dim, history_seq_len=args.history_seq_len,
                                     horizon= args.future_seq_len, num_layers=args.num_layers, default_graph=True, embed_dim=args.embed_dim, cheb_k=args.cheb_k,att_dim=args.att_dim,num_k=args.knode).cuda()


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, eps=args.epsilon)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_decay_step,
                                                                 gamma=args.lr_decay_rate)

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}



    def local_train_HINT(self, epochs):
        for epoch_num in range(epochs):
            start_time = time.time()
            self.model.train()
            total_loss = 0
            total_mae, total_rmse, total_mape = 0, 0, 0

            for batch_idx, (data, target) in enumerate(self.client_data['train_loader']):
                torch.cuda.empty_cache()
                data = data.to(self.device)  # B, T_in, N, 1
                label = target.to(self.device)  # B, T_out, N, 1
                label = label[..., 0:1]
                self.optimizer.zero_grad()
                output, global_node_embed = self.model(data)
                loss = self.loss(output, label)
                loss2 =  torch.sum(torch.mm(global_node_embed, global_node_embed.transpose(0, 1)))
                loss = loss + 0.1*loss2
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                output = self.client_data['scaler'].inverse_transform(output)
                label = self.client_data['scaler'].inverse_transform(label)

                total_mae
                total_loss += loss.item()

                output = self.client_data['scaler'].inverse_transform(output)
                label = self.client_data['scaler'].inverse_transform(label)

                total_mae += MAE_torch(output, label).item()
                total_rmse += RMSE_torch(output, label).item()
                total_mape += MAPE_torch(output, label).item()


            train_epoch_loss = total_loss / len(self.client_data['train_loader'])

            end_time = time.time()

            message1 = '  Local Epoch [{}/{}] train_loss: {:.4f},  lr: {:.6f}, {:.1f}s'.format(
                epoch_num + 1,
                epochs,  train_epoch_loss, self.optimizer.param_groups[0]['lr'],
                (end_time - start_time))

            # learning rate decay
            if self.lr_decay:
                self.lr_scheduler.step()

            val_result = self.evaluate_HINT('val')
            if val_result['mae'] < self.best_val_loss:
                self.best_val_loss = val_result['mae']
                torch.save(self.model.state_dict(), self.args.save_dir+'/Client'+str(self.client_id)+'.pth')

            test_result = self.evaluate_HINT('test')
            message2 = ("Val MAE: {:.4f}, Val MAPE: {:.4f} , Val RMSE: {:.4f};"
                        " Test MAE: {:.4f}, Test MAPE: {:.4f} , Test RMSE: {:.4f}").format(
                val_result["mae"],val_result["mape"],val_result["rmse"],
                test_result["mae"],test_result["mape"],test_result["rmse"]
            )
            self.logger.info("Client" + str(self.client_id)+ message1+"    " + message2)


    def evaluate_HINT(self, mode):
        if mode == 'val':
            dataloader = self.client_data['val_loader']
        else:
            dataloader = self.client_data['test_loader']
        with (torch.no_grad()):
            self.model.eval()
            total_val_loss = 0

            total_mae, total_rmse, total_mape,total_masked_mae,total_masked_mape,total_masked_rmse = 0, 0, 0,0,0,0
            for batch_idx, (data, target) in enumerate(dataloader):
                torch.cuda.empty_cache()
                data = data.to(self.device)
                label = target.to(self.device)
                label = label[..., 0:1]
                output, global_node_embed = self.model(data)

                loss = self.loss(output, label)

                loss2 =  torch.sum((F.relu(torch.mm(global_node_embed, global_node_embed.transpose(0, 1)))))
                loss = loss + 0.1*loss2

                total_val_loss += loss.item()

                output = self.client_data['scaler'].inverse_transform(output)
                label = self.client_data['scaler'].inverse_transform(label)

                total_mae += MAE_torch(output, label).item()
                total_rmse += RMSE_torch(output, label).item()
                total_mape += MAPE_torch(output, label).item()


            mae = total_mae / len(dataloader)
            rmse = total_rmse / len(dataloader)
            mape = total_mape / len(dataloader)

            val_loss = total_val_loss / len(dataloader)

            result = {"mae": mae,
                      "mape": mape,
                      "rmse": rmse,
                    "loss": val_loss}
            return result
