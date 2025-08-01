import sys
import os
import pickle
import copy
import operator
from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as  F
from tqdm import tqdm
import gc
sys.path.append("..")
import numpy as np
from utils.utils import find_closest_vector, find_top_k_closest_vectors, jaccard_similarity
from utils.retrieval import ModalityRetrival
from utils.config import parse_config
from datasets.mimic import MimicMultiModal, MimicPublic
from datasets.iu_xray import IUXrayMultiModal
from datasets.chexpert import chexpertdata
import torch.nn as nn
from networks import get_mmclf, EncoderResNet, EncoderBert
from transformers import AutoModel,AutoProcessor
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC, MulticlassAUROC, MultilabelPrecisionRecallCurve, MultilabelAccuracy, MultilabelF1Score, MultilabelAveragePrecision
# from models.vilt.modeling_vilt import ViltModel
# from models.vilt.image_processing_vilt import ViltImageProcessor
import math
import csv
import random
# from models.RAGPT import RAGPT

# T_max =100, init_lr = 6e-4 min_lr =8e-5

T_max = 80
init_lr = 1e-3
min_lr = 5e-4
cntlr = 0

disease_dict = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Enlarged Cardiomediastinum": 4,
    "Fracture": 5,
    "Lung Lesion": 6,
    "Lung Opacity": 7,
    "No Finding": 8,
    "Pleural Effusion": 9,
    "Pleural Other": 10,
    "Pneumonia": 11,
    "Pneumothorax": 12,
    "Support Devices": 13
}

disease_list = [
            "Atelectasis","Cardiomegaly","Consolidation","Edema",
            "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
            "Lung Opacity","No Finding","Pleural Effusion",
            "Pleural Other","Pneumonia","Pneumothorax","Support Devices"
        ]


torch.cuda.set_device(0)
evaluator = MetricCollection({
    "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None).cuda() ,
    "AUCperLabel":  MultilabelAUROC(num_labels=14, average="none", thresholds=None).cuda(),
    "PRAUC": MultilabelPrecisionRecallCurve(num_labels = 14).cuda(),
    "ACC": MultilabelAccuracy(num_labels=14, average="macro", threshold=0.5).cuda(),
    "MultilabelF1Score" : MultilabelF1Score(num_labels=14, average ="macro").cuda()
})

evaluatorper = MetricCollection({
    "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None).cuda() ,
    "AUCperLabel":  MultilabelAUROC(num_labels=14, average="none", thresholds=None).cuda(),
    "PRAUC": MultilabelPrecisionRecallCurve(num_labels = 14).cuda(),
    "ACC": MultilabelAccuracy(num_labels=14, average="macro", threshold=0.5).cuda(),
    "MultilabelF1Score" : MultilabelF1Score(num_labels=14, average ="macro").cuda()
})

class ClassificationTrainerRAG1():
    def __init__(self, args, client_id, config_path, modality, logger, dataname):
        self.modality = modality
        self.args=args
        self.cls = 3
        self.logger = logger
        self.dataname = dataname
        self.client_id=client_id
        self.config_path = config_path
        self.config = parse_config(config_path)
        self.lr = 8e-4
        self.local_epoch = 0
        self.cnt = 0
        self.auc = 0.0
        self.perauc = 0.0
        self.loss = 0.0
        self.val_track = []
        self.round = 0
        self.setup_rag()

        # self.mappings = torch.load(f"./CARckpts/gptf/{self.dataname}_map_{self.client_id}.pth")
        # self.mappings1 =  torch.load(f"./CARckpts/gptf/{self.dataname}_maptest_{self.client_id}.pth")
        # self.rdata = torch.load(f"./CARckpts/gptf/{self.dataname}_rdataidx.pth")
        # self.rdatatest = torch.load(f"./CARckpts/gptf/{self.dataname}_testrdataidx.pth")

        base_path = "./CARckpts/CARfx604"
        map_path = os.path.join(base_path, f"{self.dataname}_map_{self.client_id}.pth")
        maptest_path = os.path.join(base_path, f"{self.dataname}_maptest_{self.client_id}.pth")
        rdata_path = os.path.join(base_path, f"{self.dataname}_{self.client_id}_rdataidx.pth")
        rdatatest_path = os.path.join(base_path, f"{self.dataname}_{self.client_id}_testrdataidx.pth")
        self.mappings = torch.load(map_path) if os.path.exists(map_path) else {}
        self.mappings1 = torch.load(maptest_path) if os.path.exists(maptest_path) else {}
        self.rdata = torch.load(rdata_path) if os.path.exists(rdata_path) else {}
        self.rdatatest = torch.load(rdatatest_path) if os.path.exists(rdatatest_path) else {}

        self.logger.info(f"--{self.dataname}---{self.modality}----inilr {init_lr},  minlr {min_lr} --------------")

    def setup_rag(self):
        if self.dataname == "MIMIC":
            partition_path = f'./partitions/mimic-cxr_APPA_iid1_noleak.pkl'
            with open(partition_path, "rb") as f:
                d = pickle.load(f)

            train_set = MimicMultiModal(self.config.dataset.view, "train")

            client_partition = d["client"]
            train_idx = client_partition[self.client_id]["train"]
            # public_train = d["server"]
            # train_idx += public_train
            val_idx = client_partition[self.client_id]["val"]

            self.train_set = Subset(train_set, train_idx)
            self.test_set = Subset(train_set, val_idx)
          #  print("-------------------------MIMIC------------------------------")
            
        if self.dataname == "NIHopeni":
            partition_path = f"./partitions/openiu.pkl"
            with open(partition_path, "rb") as f:
                d = pickle.load(f)
            
            train_set = IUXrayMultiModal(self.config.dataset.view, "train")
            # client_partition = data_partition["client"]
            train_idx = d["train"]
            test_idx = d["test"]
            
            self.train_set = Subset(train_set, train_idx)
            self.test_set = Subset(train_set, test_idx)
          #  print("-------------NIHOPENIU----------------")
        
        if self.dataname == "chexpert":
            partition_path = "./partitions/chexpert_frontal_iid1_8.pkl"
            with open(partition_path, "rb") as f:
                d = pickle.load(f)
            
            train_set = chexpertdata(self.config.dataset.view, "train")
            #client_partition = d["client"]
            #train_idx = client_partition[self.client_id-self.args.num_clients]["train"]
            train_idx = d["train"]
            test_idx = d["test"]

            self.train_set = Subset(train_set, train_idx)
            self.test_set = Subset(train_set, test_idx)
          #  print("-------------chexpert----------------")

        print(f"len train {len(self.train_set)}")
        print(f"len test {len(self.test_set)}")
        print(f"---------{self.dataname}----------{self.modality}------------")
        self.train_loader = DataLoader(self.train_set, batch_size=20, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(self.test_set, batch_size=20, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)


        # partition_path = f'./partitions/mimic-cxr_APPA_iid1_noleak.pkl'
        train_set = MimicPublic("train")
        self.public_train_dset = train_set

    def generate_RAG_mapping(self, global_vec, global_labels, model, comms):
        model.eval()
        model.cuda()
        local_img_vec = []
        local_txt_vec = []
        local_idx = []
        local_label = []
        print("------------------RAGMAPPING------------Retriving Top K datasets")
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for data in tepoch:
                frames = data["image"].cuda()
                label = data["label"].cuda()
                text = data["text"]
                idx = data["idx"]
                gt = data["label"]

                with torch.no_grad():
                    if self.args.method == "rapcc" or self.args.method == "ragpt":
                        output = model(frames, text, None,None,None, "ret")
                    elif self.args.method == "CAR":
                        output = model(frames, text, "", self.modality)
                    local_img_vec.extend(output["imagefeat"].cpu().numpy())
                    local_txt_vec.extend(output["textfeat"].cpu().numpy())

                local_idx.extend(idx.numpy())
                local_label.extend(gt)

        if self.modality == "text":
            if self.args.use_refinement:
                print("----------Label Refining-------------")
                top_k_closet_idx, _ = find_top_k_closest_vectors(local_txt_vec, global_vec, 10)
            else:
                top_k_closet_idx, _, avgdis = find_closest_vector(local_txt_vec, global_vec)
        else:
            if self.args.use_refinement:
                print("----------Label Refining-------------")
                top_k_closet_idx, _ = find_top_k_closest_vectors(local_img_vec, global_vec, 10)
            else:
                top_k_closet_idx, _, avgdis = find_closest_vector(local_img_vec, global_vec)

        closet_idx = top_k_closet_idx

        self.mappings = {a:b for (a,b) in zip(local_idx, closet_idx)}

        # if self.args.method == "rapcc":
        #     torch.save(self.mappings, f"./CARckpts/gptf/{self.dataname}_map_{self.client_id}.pth")

        # with tqdm(self.test_loader, unit="batch") as tepoch:
        #     for data in tepoch:
        #         frames = data["image"].cuda()
        #         label = data["label"].cuda()
        #         text = data["text"]
        #         idx = data["idx"]
        #         gt = data["label"]

        #         with torch.no_grad():
        #             if self.args.method == "rapcc" or self.args.method == "ragpt":
        #                 output = model(frames, text, None,None,None, "ret")
        #             elif self.args.method == "CAR":
        #                 output = model(frames, text, "", self.modality)
        #             local_img_vec.extend(output["imagefeat"].cpu().numpy())
        #             local_txt_vec.extend(output["textfeat"].cpu().numpy())

        #         local_idx.extend(idx.numpy())
        #         local_label.extend(gt)

        # if self.modality == "text":
        #     if self.args.use_refinement:
        #         print("----------Label Refining-------------")
        #         top_k_closet_idx, _ = find_top_k_closest_vectors(local_txt_vec, global_vec, 10)
        #     else:
        #         top_k_closet_idx, _, avgdis = find_closest_vector(local_txt_vec, global_vec)
        # else:
        #     if self.args.use_refinement:
        #         print("----------Label Refining-------------")
        #         top_k_closet_idx, _ = find_top_k_closest_vectors(local_img_vec, global_vec, 10)
        #     else:
        #         top_k_closet_idx, _, avgdis = find_closest_vector(local_img_vec, global_vec)
        
        # closet_idx = top_k_closet_idx
        # self.mappings1 = {a:b for (a,b) in zip(local_idx, closet_idx)}
        model.cpu()

    def retrive_data1(self, local_idxs, global_idxs, mod):
        global_idx = [0]
        if self.round == 0:
            map_idx = operator.itemgetter(*local_idxs)(self.mappings)
            global_idx = operator.itemgetter(*map_idx)(global_idxs)
            if self.cnt not in self.rdata:
                self.rdata[self.cnt] = {}
            self.rdata[self.cnt] = global_idx

        else:
            global_idx=self.rdata[self.cnt]

        # if mod == "train":
        #     global_idx = self.rdata[self.cnt]
        # else:
        #     global_idx = self.rdatatest[self.cnt]      

        retrived_txt = []
        retrived_img = []
        rl = []
        for id in global_idx:
            data = self.public_train_dset[id]
            image = data["image"]
            text = data["text"]
            label = data["label"]
            retrived_txt.append(text)     
            retrived_img.append(image)
            rl.append(label)

        retrived_img = torch.stack(retrived_img, dim=0)
        rl = torch.stack(rl, dim=0)
        return {"img":retrived_img, "txt": retrived_txt, "rl":rl}

    def retrive_data(self, local_idxs, global_idxs, mod):
        if mod == "train":
            map_idx = operator.itemgetter(*local_idxs)(self.mappings)
        else:
            map_idx = operator.itemgetter(*local_idxs)(self.mappings1)

        global_idx = operator.itemgetter(*map_idx)(global_idxs)
        retrived_txt = []
        retrived_img = []
        for id in global_idx:
            data = self.public_train_dset[id]
            image = data["image"]
            text = data["text"]
            label = data["label"]
            retrived_txt.append(text)     
            retrived_img.append(image)

        retrived_img = torch.stack(retrived_img, dim=0)
        return {"img":retrived_img, "txt": retrived_txt}
    
    def run(self, comms, img_vec, txt_vec, labels, idxs, gmodel, mmmodel, imgmodel, txtmodel):
        print(f"client_id : {self.client_id} training started")
        model = copy.deepcopy(gmodel)
        model = model.cuda()
        permodel = "a"

        self.round = comms
        cos_inner = math.pi * comms / T_max
        self.lr = min_lr + 0.5 * (init_lr - min_lr) * (1 + math.cos(cos_inner))

        if self.args.method == "rapcc" or self.args.method =="ragpt":
            if comms == 0:
                if self.modality == "image" :
                    self.generate_RAG_mapping(img_vec, labels, model, comms)
                elif self.modality == "text":
                    self.generate_RAG_mapping(txt_vec, labels, model, comms)
        
        else:
            if self.modality == "image" :
                self.generate_RAG_mapping(img_vec, labels, gmodel, comms)
            elif self.modality == "text":
                self.generate_RAG_mapping(txt_vec, labels, gmodel, comms)

        for i in range(1):
            self.traineta(idxs, model, mmmodel, imgmodel, txtmodel )
            # self.testeta(idxs, model)

            print(f"Client_id: {self.client_id} local_epoch{self.local_epoch} communication round: {comms} round_epoch: {i}")
            self.local_epoch += 1
            
        model.cpu()   
        torch.save(model, f"./CARckpts/TmpMAVG1/{self.args.prefix}_{self.client_id}.pth") 

        if self.args.method == "rapcc":
            # permodel.cpu()
            # torch.save(permodel, f"./CARckpts/TmpMAVG1/{self.args.prefix}_{self.client_id}_p.pth") 
        
            if comms == 0:
                if not os.path.exists(f"./CARckpts/CARfx604/{self.dataname}_{self.client_id}_rdataidx.pth"):
                    torch.save(self.rdata, f"./CARckpts/CARfx604/{self.dataname}_{self.client_id}_rdataidx.pth")
                    print(f" save ./CARckpts/CARfx604/{self.dataname}_{self.client_id}_rdataidx.pth")
                # if not os.path.exists(f"./CARckpts/CARfx/{self.dataname}_{self.client_id}_testrdataidx.pth"):
                #     torch.save(self.rdatatest, f"./CARckpts/CARfx/{self.dataname}_{self.client_id}_testrdataidx.pth")
                #     print(f" save ./CARckpts/CARfx/{self.dataname}_{self.client_id}_rdataidx.pth")


        torch.cuda.empty_cache()
        import gc
        gc.collect()      
        

    def run1(self, comms, idxs, gmodel):
        print(f"client_id : {self.client_id} training started")
        model = copy.deepcopy(gmodel)
        model = model.cuda()
        permodel = "a"

        self.round = comms
        cos_inner = math.pi * comms / T_max
        self.lr = min_lr + 0.5 * (init_lr - min_lr) * (1 + math.cos(cos_inner))

        for i in range(1):
            self.traineta(idxs, model)
            # self.testeta(idxs, model)

            print(f"Client_id: {self.client_id} local_epoch{self.local_epoch} communication round: {comms} round_epoch: {i}")
            self.local_epoch += 1
            
        model.cpu()   
        torch.save(model, f"./CARckpts/TmpMAVG1/{self.args.prefix}_{self.client_id}.pth") 


    def traineta(self, global_idxs, model, mmmodel, imgmodel, txtmodel):
        model.train()
        model.cuda()

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()) , lr=self.lr)
        grad_scaler =  torch.cuda.amp.GradScaler()
        tmploss = 0.0
        self.cnt = 0
        imgcnt = 0
        tmpmod = "mm"

        with tqdm(self.train_loader, unit="batch") as tepoch:
            for data in tepoch:
                # tmpmod = "image"
                # if self.cnt < uni:
                #     tmpmod = "mm"
                # else:
                #     imgcnt +=1

                frames = data["image"].cuda()
                label = data["label"].cuda()
                text = data["text"]
                idx = data["idx"].tolist()
                optimizer.zero_grad()               

                if self.args.method == "CAR":
                    if self.modality == 'text':
                        out = self.retrive_data(idx, global_idxs, "train")
                        frames = out["image"].to(frames.dtype).cuda()
                    if self.modality == 'image':
                        out = self.retrive_data(idx, global_idxs, "train")
                        text = out['txt']

                    # with torch.autocast(device_type="cuda" , enabled=False):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(frames, text, "" , self.modality)
                        loss = criterion(output["logits"], label)
                        tmploss += loss.item()

                    grad_scaler.scale(loss).backward()
                    if self.config.train.grad_clip > 0:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(),self.config.train.grad_clip)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    tepoch.set_postfix(Loss=loss.item())
                
                elif self.args.method == "rapcc":
                    if self.modality != "mm":
                        rd = self.retrive_data1(idx, global_idxs, "train")
                        rframes = rd["img"].to(frames.dtype)
                        rtext = rd['txt']
                        rl=rd["rl"]
                        rframes = rframes.cuda()
                        rl = rl.cuda()
                    else:
                        rframes = 'a'
                        rtext = 'a'
                        rl = 'a'

                    output = model(frames,text,rframes,rtext,rl, self.modality) 
                    loss = criterion(output["logits"], label) 
                    tmploss += loss.item()

                    grad_scaler.scale(loss).backward()
                    if self.config.train.grad_clip > 0:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(),self.config.train.grad_clip)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    tepoch.set_postfix(Loss=loss.item())

                elif self.args.method == "ragpt":
                    rd = self.retrive_data1(idx, global_idxs, "train")
                    rframes = rd["img"].to(frames.dtype)
                    rtext = rd['txt']
                    rl=rd["rl"]
                    rframes = rframes.cuda()
                    rl = rl.cuda()
                    output = model(frames,text,rframes,rtext,rl, self.modality) 
                    loss = criterion(output["logits"], label) 
                    tmploss += loss.item()

                    loss.backward()
                    optimizer.step()   
                    tepoch.set_postfix(Loss=loss.item())

                elif self.args.method == "momke":
                    output = model(frames, text, first_stage=False, test_condition = self.modality) 
                    loss = criterion(output["logits"], label) 
                    tmploss += loss.item()

                    grad_scaler.scale(loss).backward()

                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(),2.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    tepoch.set_postfix(Loss=loss.item())

                self.cnt+=1

        tmploss /= 1.0*len(self.train_loader)
        imgcnt /= 1.0*len(self.train_loader)
        
        reg = 0.0
        self.loss = tmploss 
        self.logger.info(f"Client {self.client_id}, train tmploss {tmploss:.4f}, imgcnt {imgcnt:.3f}")
        print(f"Client {self.client_id},train tmploss {tmploss:.4f}, imgcnt {imgcnt:.3f}")

    def testeta(self, global_idxs, model):
        model.eval()
        model.cuda()
        self.cnt = 0
        evaluator.reset()

        tmpmod = "mm"
        with tqdm(self.test_loader, unit="batch") as tepoch:
            for data in tepoch:
                # tmpmod = "image"
                # if self.cnt < uni:
                #     tmpmod = "mm"
                # else:
                #     imgcnt += 1

                frames = data["image"].cuda()
                label = data["label"].cuda()
                text = data["text"]
                idx = data["idx"].cpu().numpy()

                if self.args.method == "CAR":
                    ret = self.retrive_data(idx, self.idxs, "test")
                    rtext = ret["txt"]
                    rframes =  ret["img"].to(frames.dtype).cuda()
                    rl = ret["rl"].cuda()
                elif self.args.method =="ragpt" or self.args.method =="rapcc":
                    rd = self.retrive_data1(idx, global_idxs, "test")
                    rframes = rd["img"].to(frames.dtype)
                    rtext = rd['txt']
                    rl=rd["rl"]
                    rframes = rframes.cuda()
                    rl = rl.cuda()

                self.cnt += 1
                with torch.no_grad():
                    if self.args.method == "rapcc" or self.args.method == "ragpt":
                        server_logits = model(frames, text, rframes, rtext, rl, self.modality)['logits']
                    elif self.args.method == "CAR":
                        if self.modality == "image":
                            server_logits = model(frames, rtext, "", self.modality)['logits']
                        else:
                            server_logits = model(frames, text, "", self.modality)['logits']
                    elif self.args.method == "momke":
                        server_logits = model(frames, text, first_stage=False, test_condition = self.modality)['logits']                     
                evaluator.update(server_logits.sigmoid(), label.long())

        metrics = evaluator.compute()
        print( f" round: {self.round},test AUC(Aggregrated): {metrics['AUC'].item():.5f}, test ACC {metrics['ACC'].item():.5f} ")
        self.logger.info(f"round: {self.round} ,Test AUC(Aggregrated): {metrics['AUC'].item():.5f}, test ACC {metrics['ACC'].item():.5f},  macroF1 {metrics['MultilabelF1Score'].item():.5f}")
