import os
import random
import numpy as np
import torch
import logging
import argparse

from algorithms.ServerTrainers import ClassificationTrainer
from algorithms.FedAvg import FedAvg
from algorithms.FedAvgIn import FedAvgIn, FedAvgInRAG
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser(description='Federated Learning')


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_loggger(log_name, log_path, log_level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler(log_path)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    return logger

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    #To let the cuDNN use the same convolution every time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_wandb(args):
    import wandb
    name = f"{str(args.name)}"

    wandb.init(
        project="qualitative",
        name = name,
        resume = None,
        config=args
    )

    return wandb

def args():
    parser.add_argument('--name', type=str, default='Test', help='The name for different experimental runs.')
    parser.add_argument('--exp_dir', type=str, default='./experiments/',
                        help='Locations to save different experimental runs.')
    parser.add_argument('--server_config_path', type=str, default='configs/server_configs.yaml',
                        help='Location for server configs')
    parser.add_argument('--client_config_path', type=str, default='configs/client_configs.yaml',
                        help='Location for client configs')
    parser.add_argument('--comm_rounds', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--algorithm', type=str, default='fedavgRAG', choices=['fedavgRAG'],
                        help='Choice of Federated Averages')
    parser.add_argument('--num_clients', type=int, default=2,
                        help='total number of multimodal clients')
    parser.add_argument('--img_clients', type=int, default=8,
                        help='total number of image clients')
    parser.add_argument('--txt_clients', type=int, default=0,
                        help='total number of text clients')
    parser.add_argument('--warmup', type=int, default=30,
                        help='total number of text clients')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='noise_level')
    parser.add_argument('--save_clients', action="store_true", default=False)
    parser.add_argument('--use_refinement', action="store_true", default=False)
    parser.add_argument('--gpu', type=int, default=0,)
    parser.add_argument('--loginfo', type=str, default="loginfo",)
    parser.add_argument('--logabs', type=str, default="logabs",)
    parser.add_argument('--prefix', type=str, default="logabs",)
    parser.add_argument('--isseg', type=str, default="notseg",)
    parser.add_argument('--method', type=str, default="notori",)

    
args()
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    wandb = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    gpux = args.gpu
    loginfo = args.loginfo
    logabstr = args.logabs
    torch.cuda.set_device(gpux)
    logger = setup_loggger('info', loginfo)
    logger.info(f"gpu   {gpux}")
    logger.info(f"{args.prefix}")

    if args.algorithm == 'fedavgRAG':
        engine = FedAvgInRAG(args, logger, wandb)
        engine.run()

    else:
        raise ValueError(f"Not implemented {args.algorithm}")
