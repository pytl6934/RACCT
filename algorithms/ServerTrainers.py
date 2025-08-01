import sys
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
sys.path.append("..")
from utils.config import parse_config
from datasets.mimic import MimicMultiModal
from datasets.iu_xray import IUXrayMultiModal

from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC, MulticlassAUROC
from models import ViltModel, ViltImageProcessor,RACCT
from transformers import BertTokenizer
from peft import get_peft_model, LoraConfig, TaskType


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)

class ClassificationTrainer:
    def __init__(self, args, config_path, dataname , logger, isRAG , wandb=False):
        self.args = args
        self.logger = logger
        self.wandb = wandb
        self.isRAG = isRAG
        self.dataname = dataname
        self.config = parse_config(config_path)
        self.dset_name = self.config.dataset.dset_name
        self.cls = 3
        self.model = None
        self.load_data()
        self.load_model()

        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
        })

        self.cur_epoch = 0
        self.save_dir = os.path.join(self.args.exp_dir, "server")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.val_track = []

    def load_data(self):
        if self.dataname == "MIMIC":
            partition_path = f'partitions/mimic-cxr_APPA_iid1_noleak.pkl'
            with open(partition_path, "rb") as f:
                data_partition = pickle.load(f)
            # train_set = MimicMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "train")
            # train_idx = data_partition["server"]
            # self.train_set = Subset(train_set, train_idx)
            self.val_set = MimicMultiModal(self.config.dataset.view, "val")
            self.test_set = MimicMultiModal(self.config.dataset.view, "test")
        
        if self.dataname == "NIHopeni":
            partition_path = f"./partitions/iuxray_frontal_iid1_noleak.pkl"
            with open(partition_path, "rb") as f:
                data_partition = pickle.load(f)
            
            train_set = IUXrayMultiModal(self.config.dataset.view, "train")
            client_partition = data_partition["client"]
            train_idx = client_partition[0]["val"]

            # public_train = data_partition["server"]
            # train_idx += public_train

            # val_idx = client_partition[self.client_id]["val"]
            self.train_set = Subset(train_set, train_idx)
            # self.val_set = Subset(train_set, val_idx)
            print("-------------server    NIHOPENIU----------------")


        # self.train_loader = DataLoader(self.train_set, batch_size=self.config.dataloader.batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        self.val_loader = DataLoader(self.val_set, batch_size=5, shuffle=False, num_workers= 8, pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(self.test_set, batch_size=20, shuffle=False, num_workers= 8, pin_memory=True, drop_last=True)
        print(f"val len {len(self.val_loader.dataset)}")

        print("------------------------------server Data Loaded Successfully-------------------------")


    def load_model(self):
        if self.isRAG == "RAG":
            self.model = RACCT(            
                prompt_position = 0,
                prompt_length = 1,
                dropout_rate= 0.2,
                lbd = 0.3,
                max_text_len = 128,
                max_image_len = 145, )

            file_path = f'./CARckpts/Server/{self.args.prefix}.pth'
            if os.path.exists(file_path):
                tmp_dict = torch.load(file_path)
                self.model.load_state_dict(tmp_dict, strict=False)     
                print("------------------load model ---------------")             


        self.lr = 1e-5
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.grad_scaler =  torch.cuda.amp.GradScaler()

        print("------------------------------server Model Loaded Successfully-------------------------")

    def save_best(self, comms, logger):
        ckpt_path =f"./CARckpts/Final/{self.args.prefix}.pth"
        print(f"test round: {comms}, save server best model ")
        logger.info(f"test round: {comms}, save server best model, {self.args.prefix} ")
        torch.save(self.model.state_dict(), ckpt_path)

    def load_best(self):
        ckpt_path =f"./CARckpts/Final/{self.args.prefix}.pth"
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint)

    def save_bestRAG(self, comms, logger):
        ckpt_path =f"./CARckpts/Final/{self.args.prefix}.pth"
        print(f"test round: {comms}, save server best model ")
        logger.info(f"test round: {comms}, save server best model, {self.args.prefix}")
        torch.save(self.model.state_dict(), f"./CARckpts/Server/{self.args.prefix}.pth")

    def load_bestRAG(self):
        ckpt_path =f"./CARckpts/Server/{self.args.prefix}.pth"
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint)
    

    def save_log(self):
        log_path = os.path.join(self.save_dir, "val_aucs.pkl")
        with open(log_path, "wb") as f:
            pickle.dump(self.val_track, f)

    def run_standalone(self):
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        print("----------------Standalone Training-------------------------")
        self.val_auc = 0
        self.model.cuda()
        for i in range(2):
            print(f"Server: Epoch {i}")
            self.train_epoch()

            self.cur_epoch+=1
        print("------------------------------------------------------------")

        # self.load_best()
        # self.test()
    

    def test(self):
        self.model.cuda()
        self.model.eval()
        test_evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
            "AUCperLabel":  MultilabelAUROC(num_labels=14, average="none", thresholds=None)
        })
        with tqdm(self.test_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model( images, text)
                test_evaluator.update(output["logits"], label.long())
        metrics = test_evaluator.compute()
        print(f"AUC : {metrics['AUC']}")
        print(f"AUCperLabel : {metrics['AUCperLabel']}")
        self.wandb.log({"Test AUC(Aggregrated)":metrics['AUC'].item()})
        self.evaluator.reset()


    def run(self, comms):
        self.model.cuda()
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        for i in range(2):
            print(f"Server:-  Comm round{comms} local_epoch:{self.cur_epoch}  round_epoch: {i}")
            self.train_epoch()
            self.cur_epoch +=1
            print("------------------------------------------------------------")
        self.model.cpu()
        import gc
        gc.collect()

    def train_epoch(self):
        self.model.train()
        print("Training Model:")
        with tqdm(self.test_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(images, None)
                    loss = self.criterion(output["logits"], label)
                
                self.grad_scaler.scale(loss).backward()
                
                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)
                
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())

    def val(self):
        self.model.eval()
        print('Validating Model:')
        with tqdm(self.test_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model( images, text)
                self.evaluator.update(output["logits"], label.long())
        metrics = self.evaluator.compute()
        print(f"Val AUC : {metrics['AUC']}")
        if self.wandb:
            self.wandb.log({"Val AUC(Server)":metrics['AUC'].item()}, step=self.cur_epoch)
        self.evaluator.reset()
        return metrics['AUC'].item()


        