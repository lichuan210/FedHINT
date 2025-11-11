# Inter-Client Dependency Recovery with Hidden Global Components for Federated Traffic Prediction

PyTorch implementation of "Inter-Client Dependency Recovery with Hidden Global Components for Federated Traffic Prediction". 


## Requirement
- Python 3.10
- PyTorch 1.12.1+cu113

## Instruction
Due to file size limitations, we only provide the PEMS08 dataset in this repository. You can run the code with the following command:
```bash
python main.py ./config/AGCRN/PEMS08FLOW.json --num_client 6 --knode 32 --save_dir test
```
Additionally, we provide logs of the training process for PEMS08 with 6 clients in the following directory:
```bash
./save/FedHINT/PEMS08FLOW/6
```
