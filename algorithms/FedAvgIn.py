import sys
import copy
import os
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
from torch.nn.functional import softmax
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC, MulticlassAUROC, MultilabelPrecisionRecallCurve, MultilabelAccuracy, MultilabelF1Score, MultilabelAveragePrecision
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryAUROC,
)
import gc
from datasets.mimic import MimicPublic
from .ClientTrainers import ClassificationTrainer as ClientClassificationTrainer,  ClassificationTrainerRAG1
from .ServerTrainers import ClassificationTrainer as ServerClassificationTrainer
from utils.retrieval import ModalityRetrival
from sklearn.metrics import auc
import torch.nn.functional as F
from utils.utils import find_closest_vector, find_top_k_closest_vectors, jaccard_similarity
import operator
import random
from datasets.RSNA2 import RSNA2

class FedAvgInRAG:
    def __init__(self, args, logger, wandb):
        self.args = args
        self.wandb = wandb
        self.modality = "mm"
        self.round = 0
        self.device = torch.device("cuda")
        self.num_mm_clients = args.num_clients ## Needed from Args
        self.total_comms = args.comm_rounds ## Needed from Args
        self.num_img_clients = args.img_clients
        self.num_txt_clients = args.txt_clients
        self.num_clients = self.num_mm_clients + self.num_img_clients + self.num_txt_clients
        self.logger = logger
        self.server = ServerClassificationTrainer(self.args, self.args.server_config_path, "MIMIC", logger, "RAG", False )

        self.mappings = {}
        self.mappingsval = {}
        self.rdata = {}
        self.cur_auc = 0.0

        self.publicset = None
        self.cnt = 0
        self.rsnad = RSNA2(split = "test")
        self.rsnadl = DataLoader(self.rsnad, batch_size=20, shuffle=False, num_workers= 8, pin_memory=True, drop_last=True)
        self.rdataRSNA = {}
        self.mappingsRSNA = {}
       
        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None).cuda() ,
            "AUCperLabel":  MultilabelAUROC(num_labels=14, average="none", thresholds=None).cuda(),
            "PRAUC": MultilabelPrecisionRecallCurve(num_labels = 14).cuda(),
            "ACC": MultilabelAccuracy(num_labels=14, average="macro", threshold=0.5).cuda(),
            "MultilabelF1Score" : MultilabelF1Score(num_labels=14, average ="macro").cuda()
        })

        self.val_track = []

    def aggregate_models(self, models, weight):
        if not models:
            return None
        averaged_model = copy.deepcopy(self.server.model)
        averaged_model = averaged_model.cpu()
        averaged_model.load_state_dict({k: v.clone().cpu() * 0.0 for k, v in models[0].state_dict().items()})
        for model in models:
            model = model.cpu()
            model_params = model.state_dict()
            for name, param in averaged_model.state_dict().items():
                averaged_param = param + model_params[name] * weight
                averaged_model.state_dict()[name].copy_(averaged_param)
        num_models = len(models)
        for name, param in averaged_model.state_dict().items():
            averaged_param = param / (num_models * weight)
            averaged_model.state_dict()[name].copy_(averaged_param)
        return averaged_model

    def setup_clients1(self):
        mimic_clients = [ClassificationTrainerRAG1(self.args, 0, self.args.client_config_path, "mm" , self.logger, "MIMIC")]
        openiu_clients = [ClassificationTrainerRAG1(self.args, 1, self.args.client_config_path, "mm" , self.logger, "NIHopeni") ]
        chexpert_clients = [ClassificationTrainerRAG1(self.args, 2, self.args.client_config_path, "image", self.logger, "chexpert")]
        
        self.clients = mimic_clients + openiu_clients + chexpert_clients
        self.num_clients = len(self.clients)
        
        self.aggweight = [len(self.clients[i].train_set) for i in range(len(self.clients))]
        self.logger.info(f"client data  {self.aggweight} ")
        aggsum =  sum(self.aggweight)
        self.aggweight = [1.0 * x/aggsum for x in self.aggweight]    

    def get_public_dset(self):
        partition_path = f'partitions/mimic-cxr_APPA_iid1_noleak.pkl'
        with open(partition_path, "rb") as f:
            data_partition = pickle.load(f)
        self.publicset = MimicPublic("eval")
        self.publid_idx = train_idx = data_partition["server"]
        self.public_dset = Subset(self.publicset, train_idx)
        self.public_dset_loader = DataLoader(self.public_dset, batch_size=20, shuffle=False, num_workers= 8, pin_memory=True, drop_last=True)

    def testeta(self, eta):
        self.cnt = 0
        self.evaluator.reset()
        imgcnt = 0
        with tqdm(self.server.test_loader, unit="batch") as tepoch:
            for data in tepoch:
                tmpmod = "mm"
                x = random.random()
                if x < eta:
                    tmpmod = "image"
                    imgcnt+=1

                frames = data["image"].cuda()
                label = data["label"].cuda()
                text = data["text"]
                idx = data["idx"].cpu().numpy()

                if tmpmod != "mm" :
                    ret = self.retrive_data(idx , self.idxs)
                    rtxt = ret["txt"]
                    rimg =  ret["img"].to(frames.dtype).cuda()
                    rl = ret["rl"].cuda()

                self.cnt += 1
                with torch.no_grad():
                    if self.args.method == "racct":
                        server_logits = self.server.model(frames, text, rimg, rtxt, rl, tmpmod)['logits']
                  
                self.evaluator.update(server_logits.sigmoid(), label.long())

        metrics = self.evaluator.compute()
        imgcnt/=1.0*len(self.server.test_loader)

        print( f"{eta:.2f} round: {self.cur_comms},test AUC(Aggregrated): {metrics['AUC'].item():.5f}, test ACC {metrics['ACC'].item():.5f} ")
        self.logger.info(f"{eta:.2f} round: {self.cur_comms} ,Test AUC(Aggregrated): {metrics['AUC'].item():.5f}, test ACC {metrics['ACC'].item():.5f},  macroF1 {metrics['MultilabelF1Score'].item():.5f}")
        self.logger.info(f"imgcnt test {imgcnt:.3f}")

        if eta == -1:
            self.cur_auc = metrics['AUC'].item()

    def val(self, dataname, dataloader, evid):
        self.server.model.eval()
        self.server.model.cuda()
        print('Validating Model:')
        evalRSNA= MetricCollection({
            "AUC": BinaryAUROC().cuda(),
            "ACC": BinaryAccuracy().cuda(),
            "F1": BinaryF1Score().cuda(),
        })

        self.cnt = 0
        evalRSNA.reset()
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tmpmod = "image"
                frames = data["image"].cuda()
                label = data["label"].cuda()
                text = data["text"]
                idx = data["idx"].cpu().numpy()

                ret = self.retrive_datarsna(self, idx, self.idxs)
                rtxt = ret["txt"]
                rimg =  ret["img"].to(frames.dtype).cuda()
                rl = ret["rl"].cuda()
                self.cnt += 1
                with torch.no_grad():
                    server_logits = self.server.model(frames, text, rimg, rtxt, rl, tmpmod)['logits']
                    logits_label7 = server_logits[:, evid]  
                    targets_label7 = label[:, evid].long()

                evalRSNA.update(logits_label7.sigmoid(), targets_label7.long())

        metrics = evalRSNA.compute()
        print( f"{dataname} zeroshot round: {self.cur_comms}, val AUC: {metrics['AUC'].item():.5f}, val ACC {metrics['ACC'].item():.5f}  ")
        self.logger.info(f"{dataname} zeroshot round: {self.cur_comms} , val AUC: {metrics['AUC'].item():.5f}, val ACC {metrics['ACC'].item():.5f},  F1 {metrics['F1'].item():.5f} ")                

    def test(self):
        self.server.model.eval()
        self.server.model.cuda()

        self.testeta(-1) # 0% missing rate
        if self.round > 18:
            self.testeta(0.5) # 50% missing rate
            self.testeta(0.1) # 10% missing rate
            self.testeta(0.9)
            self.testeta(0.3)
            self.testeta(0.7)

    def aggregate(self):   
        global_dict = {k: torch.zeros_like(v, dtype=torch.float32).cpu() for k, v in self.server.model.state_dict().items()}
        for i, station in enumerate(self.clients):
            print(f"agg {station.dataname}")
            client_path = f"./ckpts/TmpMAVG1/{self.args.prefix}_{i}.pth"  
            client_model = torch.load(client_path, map_location='cpu')
            client_model_state_dict = client_model.state_dict()

            for k in global_dict.keys():
                if any(substring in k for substring in ["num_batches_tracked", "embeddings.position_ids"]):
                    continue  
                para = client_model_state_dict.get(k)  
                if para is not None:
                    global_dict[k] += para.float() * self.aggweight[i]  

        self.server.model.load_state_dict(global_dict)
        del global_dict
        torch.cuda.empty_cache()
        gc.collect()

    def run(self):
        self.cur_comms = 0
        self.val_auc = 0
        self.get_public_dset()

        self.setup_rag()
        self.setup_clients1()
        self.num_clients = len(self.clients)        
        
        for comms in range(0, 1000, 1):
            tmploss = 0.0
            self.round = comms
            print(f"-----------------------Communication Round: {comms}-------------------------------")
            print(f"-----------------------Training Client Models in Clients Data------------------------------")
            self.logger.info(f"-----------------------Communication Round: {comms}-------------------------------")
            for i in range(len(self.clients)):
                self.clients[i].run(comms, self.img_vec, self.txt_vec, self.labels, self.idxs, self.server.model)
                tmploss += self.clients[i].loss       
            
            tmploss /= 1.0*len(self.clients)
            print(f"avgloss {tmploss:.4f}")
            self.logger.info(f"avgloss {tmploss:.4f}")
            print(f"-----------Performing global aggregration--------------------------")
           
            self.aggregate()
            
            print("---------------Evaluating Aggregrated Model in Val Set-------------------------------")
            if self.round > 8:
                self.val("RSNA", self.rsnadl, 7)
            
            self.test()
            self.val_track.append(self.cur_auc)
            if self.cur_auc > self.val_auc:
                self.val_auc = self.cur_auc
                self.server.save_bestRAG(self.cur_comms, self.logger)
            self.cur_comms +=1
            gc.collect()
        self.server.load_bestRAG()
    
    def setup_rag(self):
        img_vec = []
        txt_vec = []
        idxs = []
        labels = []
        self.server.model.cuda()
        self.server.model.eval()

        print("---Genearting global data features ----------")
        with tqdm(self.public_dset_loader, unit="batch") as tepoch:
            # for frame, gt, report, prompt, idx in tepoch:
            for data in tepoch:
                frames = data["image"].cuda()
                label = data["label"].cuda()
                text = data["text"]
                idx = data["idx"]
                gt = data["label"]
                with torch.no_grad():
                    if self.args.method == "racct":
                        output = self.server.model(frames, text, None, None, None, "ret")
                    feat_img = output['imagefeat']
                    feat_text = output['textfeat']

                img_vec.extend(feat_img.cpu().numpy())
                txt_vec.extend(feat_text.cpu().numpy())
                idxs.extend(idx)
                labels.extend(gt)
        self.server.model.cpu()
        print(f"Total {len(img_vec)} features generated")
        print(f"Total {len(txt_vec)} features generated")
        self.img_vec = img_vec
        self.txt_vec = txt_vec
        self.idxs = idxs
        self.labels = labels
        
        self.mappings = self.generate_RAG_mapping(self.server.test_loader, self.img_vec, self.labels, self.server.model, self.round)
        self.mappingsval = self.generate_RAG_mapping(self.server.val_loader, self.img_vec, self.labels, self.server.model, self.round)
        self.mappingsrsna = self.generate_RAG_mapping(self.rsnadl, self.img_vec, self.labels, self.server.model, self.round)
        

    def generate_RAG_mapping(self, loader, global_vec, global_labels, model, comms):
        model.eval()
        model.cuda()
        local_img_vec = []
        local_txt_vec = []
        local_idx = []
        local_label = []
        print("------------------Server testset RAGMAPPING------------Retriving Top K datasets")
        with tqdm(loader, unit="batch") as tepoch:
            for data in tepoch:
                frames = data["image"].cuda()
                text = data["text"]
                idx = data["idx"]
                gt = data["label"]

                with torch.no_grad():
                    if self.args.method == "racct":
                        output = model(frames, text, None,None,None, "ret")

                    local_img_vec.extend(output["imagefeat"].cpu().numpy())
                    local_txt_vec.extend(output["textfeat"].cpu().numpy())

                local_idx.extend(idx.numpy())
                local_label.extend(gt)

        if self.modality == "text":
            if self.args.use_refinement:
                print("----------Label Refining-------------")
                top_k_closet_idx, _ = find_top_k_closest_vectors(local_txt_vec, global_vec, 10)
            else:
                top_k_closet_idx, _, avgdis = find_closest_vector(local_txt_vec, self.img_vec)
        else:
            if self.args.use_refinement:
                print("----------Label Refining-------------")
                top_k_closet_idx, _ = find_top_k_closest_vectors(local_img_vec, global_vec, 10)
            else:
                top_k_closet_idx, _, avgdis = find_closest_vector(local_img_vec, self.img_vec)
        
        if self.args.use_refinement:
            closet_idx = []
            print("Refining the retrived data")
            for i in tqdm(range(len(top_k_closet_idx))):
                cur_label = local_label[i]
                top_ks = top_k_closet_idx[i]
                global_label = operator.itemgetter(*top_ks)(global_labels)
                similarities = [jaccard_similarity(cur_label.numpy(), label.numpy()) for label in global_label]
                closest_label_index = torch.argmax(torch.tensor(similarities)).item()
                closet_idx.append(top_ks[closest_label_index])
        else:
            closet_idx = top_k_closet_idx

        mapping = {a:b for (a,b) in zip(local_idx, closet_idx)}
        model.cpu()
        return mapping


    def retrive_data(self, local_idxs, global_idxs):
        global_idx = [0]
        if self.round == 0:
            map_idx = operator.itemgetter(*local_idxs)(self.mappings) # or self.mappings1
            global_idx = operator.itemgetter(*map_idx)(global_idxs)
            if self.cnt not in self.rdata:
                self.rdata[self.cnt] = {}
            self.rdata[self.cnt] = global_idx

        else:
            global_idx = self.rdata[self.cnt]

        #  global_idx = self.rdata[self.cnt]  you can save self.rdata as a map and load it when you run this next time
        retrived_data = []
        retrived_data1 = []
        rl = []
        for id in global_idx:
            data = self.publicset[id]
            image = data["image"]
            text = data["text"]
            gt = data["label"]
            retrived_data.append(image)
            retrived_data1.append(text)  
            rl.append(gt)
            
        retrived_data = torch.stack(retrived_data, dim=0)
        rl = torch.stack(rl, dim=0)

        return {"img":retrived_data , "txt":retrived_data1, "rl":rl}


    def retrive_datarsna(self, local_idxs, global_idxs):
        global_idx = [0]

        if self.round == 0:
            map_idx = operator.itemgetter(*local_idxs)(self.mappingsrsna)
            global_idx = operator.itemgetter(*map_idx)(global_idxs)
            if self.cnt not in self.rdata:
                self.rdatapneu[self.cnt] = {}
            self.rdatapneu[self.cnt] = global_idx

        else:
            global_idx = self.rdataRSNA[self.cnt]

        # global_idx = self.rdataRSNA[self.cnt]  you can save self.rdata as a map and load it when you run this next time
        retrived_data = []
        retrived_data1 = []
        rl = []
        for id in global_idx:
            data = self.publicset[id]
            image = data["image"]
            text = data["text"]
            gt = data["label"]
            retrived_data.append(image)
            retrived_data1.append(text)  
            rl.append(gt)
            
        retrived_data = torch.stack(retrived_data, dim=0)
        rl = torch.stack(rl, dim=0)

        return {"img":retrived_data , "txt":retrived_data1, "rl":rl}

  
