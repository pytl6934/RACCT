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
        self.round =0
        self.device = torch.device("cuda")
        self.num_mm_clients = args.num_clients ## Needed from Args
        self.total_comms = args.comm_rounds ## Needed from Args
        self.num_img_clients = args.img_clients
        self.num_txt_clients = args.txt_clients
        self.num_clients = self.num_mm_clients + self.num_img_clients + self.num_txt_clients
        self.logger = logger
        self.server = ServerClassificationTrainer(self.args, self.args.server_config_path, "MIMIC", logger, "RAG", False )
        self.warmup = 30
        self.mappings = {}
        self.mappings1 = torch.load(f"./CARckpts/Server/mimictestMap.pth")
        self.rdata = torch.load("./CARckpts/Server/mimictestrdata.pth")
        self.cur_auc = 0.0

        self.publicset = None
        self.cnt = 0
        self.rsnad = RSNA2(split = "test")
        self.rsnadl = DataLoader(self.rsnad, batch_size=20, shuffle=False, num_workers= 8, pin_memory=True, drop_last=True)
        self.rdataRSNA = torch.load("./CARckpts/gptf/RSNA2ridx.pth")
        self.mappingsRSNA = {}
        self.cxpneu = ChestXRayTestDataset()
        self.cxpneudl = DataLoader(self.cxpneu, batch_size=20, shuffle=False, num_workers= 8, pin_memory=True, drop_last=True)
        self.mappingscxpneu = {}
        self.rdatapneu = torch.load(f"./CARckpts/gptf/cxpneuridx.pth")
        self.mmmodel = copy.deepcopy(self.server.model)
        self.imgmodel=copy.deepcopy(self.server.model)
        self.txtmodel=copy.deepcopy(self.server.model)

        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None).cuda() ,
            "AUCperLabel":  MultilabelAUROC(num_labels=14, average="none", thresholds=None).cuda(),
            "PRAUC": MultilabelPrecisionRecallCurve(num_labels = 14).cuda(),
            "ACC": MultilabelAccuracy(num_labels=14, average="macro", threshold=0.5).cuda(),
            "MultilabelF1Score" : MultilabelF1Score(num_labels=14, average ="macro").cuda()
        })

        self.val_track = []

    def setup_clients1(self):
        # openiu_clients = [ClassificationTrainerRAG1(self.args, 0, self.args.client_config_path, "mm" , self.logger, "NIHopeni") ]
        # mimic_clients = [ClassificationTrainerRAG1(self.args, 1, self.args.client_config_path, "mm" , self.logger, "MIMIC")]
        
        # chexpert_clients = [ClassificationTrainerRAG1(self.args, 2, self.args.client_config_path, "image", self.logger, "chexpert")]
        # NIHcxr = [ClassificationTrainerRAG1(self.args, 3, self.args.client_config_path, "image", self.logger, "NIHcxr")]
        # openiu_clients + mimic_clients
        # self.clients =  openiu_clients + mimic_clients 

        mm_clients = [ClassificationTrainerRAG1(self.args, i,self.args.client_config_path, "mm", self.logger, "MIMIC") for i in range(self.num_mm_clients)]
        img_clients = [ClassificationTrainerRAG1(self.args, i+self.num_mm_clients, self.args.client_config_path,  "image", self.logger, "MIMIC") for i in range(self.num_img_clients)]
        txt_clients = [ClassificationTrainerRAG1(self.args, i+self.num_mm_clients + self.num_img_clients,self.args.client_config_path, "text", self.logger, "MIMIC") for i in range(self.num_txt_clients)]
        self.clients = mm_clients + img_clients + txt_clients        
        
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

                if self.args.method != "momke":
                    ret = self.retrive_data( "test")
                    rtxt = ret["txt"]
                    rimg =  ret["img"].to(frames.dtype).cuda()
                    rl = ret["rl"].cuda()

                self.cnt += 1
                with torch.no_grad():
                    if self.args.method == "rapcc" or self.args.method == "ragpt":
                        server_logits = self.server.model(frames, text, rimg, rtxt, rl, tmpmod)['logits']
                    elif self.args.method == "CAR":
                        if tmpmod == "image":
                            server_logits = self.server.model(frames, rtxt, "", tmpmod)['logits']
                        else:
                            server_logits = self.server.model(frames, text, "", tmpmod)['logits']
                    elif self.args.method == "momke":
                        server_logits = self.server.model(frames, text, first_stage=False, test_condition = tmpmod)['logits']                     
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

                ret = self.retrive_datarsna()
                rtxt = ret["txt"]
                rimg =  ret["img"].to(frames.dtype).cuda()
                rl = ret["rl"].cuda()
                self.cnt += 1
                with torch.no_grad():
                    if self.args.method == "rapcc" or self.args.method == "ragpt":
                        server_logits = self.server.model(frames, text, rimg, rtxt, rl, tmpmod)['logits']
                    elif self.args.method == "CAR":
                        if tmpmod == "image":
                            server_logits = self.server.model(frames, rtxt, "", tmpmod)['logits']
                        else:
                            server_logits = self.server.model(frames, text, "", tmpmod)['logits']
                    elif self.args.method == "momke":
                        server_logits = self.server.model(frames, text, first_stage=False, test_condition = tmpmod)['logits'] 

                    logits_label7 = server_logits[:, evid]  
                    targets_label7 = label[:, evid].long()

                evalRSNA.update(logits_label7.sigmoid(), targets_label7.long())

        metrics = evalRSNA.compute()
        print( f"{dataname} zeroshot round: {self.cur_comms}, val AUC: {metrics['AUC'].item():.5f}, val ACC {metrics['ACC'].item():.5f}  ")
        self.logger.info(f"{dataname} zeroshot round: {self.cur_comms} , val AUC: {metrics['AUC'].item():.5f}, val ACC {metrics['ACC'].item():.5f},  F1 {metrics['F1'].item():.5f} ")                


    def test(self):
        self.server.model.eval()
        self.server.model.cuda()

        self.testeta(-1)
        self.testeta(0.5)
        if self.round > 15:
            self.testeta(0.1)
            self.testeta(0.9)
            self.testeta(0.3)
            self.testeta(0.7)
            self.testeta(1.1)

    def aggregate(self):
        global_dict = {k: torch.zeros_like(v, dtype=torch.float32).cpu() for k, v in self.server.model.state_dict().items()}
        for i, station in enumerate(self.clients):
            client_path = f"./CARckpts/TmpMAVG1/{self.args.prefix}_{i}.pth"  
            client_model_state_dict = torch.load(client_path, map_location='cpu').state_dict()
            
            for k in global_dict.keys():
                if any(substring in k for substring in ["num_batches_tracked", "embeddings.position_ids"]):
                    continue  
                
                para = client_model_state_dict.get(k)  
                if para is not None:
                    global_dict[k] += para.float() * self.aggweight[i]  
        
        self.server.model.load_state_dict(global_dict)


        self.server.model = self.server.model.cpu()
        server_state_dict = self.server.model.state_dict()
        for i, station in enumerate(self.clients):
            client_path = f"./CARckpts/TmpMAVG1/{self.args.prefix}_{i}.pth"
            client_model = torch.load(client_path, map_location='cpu')
            client_state_dict = client_model.state_dict()
            # if self.round == 0:
            #     clw = 0.7
            #     glw =0.3
            # else:
            #     clw = 0.7 * self.clients[i].perauc
            #     glw = 0.3 * self.clients[i].auc
            #     sumw = clw+glw
            #     clw = 1.0 * clw/sumw
            #     glw = 1.0 * glw/sumw
            # print(f"clw {clw:.3f},  glw {glw:.3f}")

            clw = 0.7
            glw = 0.3

            new_state_dict = {}
            for k in client_state_dict.keys():
                # , "MMG" ,"fusgate"
                if any(substring in k for substring in ["num_batches_tracked", "embeddings.position_ids", "MMG"]):
                    new_state_dict[k] = client_state_dict[k]
                    continue

                if k in server_state_dict:
                    fused_param = client_state_dict[k].float() * clw + server_state_dict[k].float() * glw
                    new_state_dict[k] = fused_param
                else:
                    new_state_dict[k] = client_state_dict[k]

            client_model.load_state_dict(new_state_dict)
            torch.save(client_model, f"./CARckpts/TmpMAVG1/{self.args.prefix}_{i}_p.pth") 

        del global_dict
        torch.cuda.empty_cache()
        gc.collect()

    def aggregateCAR(self):
        mm_clients = []
        text_clients = []
        image_clients = []        

        global_dict = {k: torch.zeros_like(v, dtype=torch.float32).cpu() for k, v in self.server.model.state_dict().items()}
        for i, station in enumerate(self.clients):
            client_path = f"./CARckpts/TmpMAVG1/{self.args.prefix}_{i}.pth"  
            client_model = torch.load(client_path, map_location='cpu')
            client_model_state_dict = client_model.state_dict()
            
            # if station.modality == "mm":
            #     mm_clients.append(client_model)
            # elif station.modality == "text":
            #     text_clients.append(client_model)
            # elif station.modality == "image":
            #     image_clients.append(client_model)

            for k in global_dict.keys():
                if any(substring in k for substring in ["num_batches_tracked", "embeddings.position_ids"]):
                    continue  
                para = client_model_state_dict.get(k)  
                if para is not None:
                    global_dict[k] += para.float() * self.aggweight[i]  

        self.server.model.load_state_dict(global_dict)

        # mm_weight = 1 / len(mm_clients) if mm_clients else 0
        # img_weight = 1 / len(image_clients) if image_clients else 0
        # txt_weight = 1 / len(text_clients) if text_clients else 0

        # mm_aggregated_model = self.aggregate_models(mm_clients, mm_weight)
        # img_aggregated_model = self.aggregate_models(image_clients, img_weight)
        # txt_aggregated_model = self.aggregate_models(text_clients, txt_weight)

        # self.mmmodel.load_state_dict(mm_aggregated_model.state_dict()) 
        # if len(image_clients)!=0:
        #     self.imgmodel.load_state_dict(img_aggregated_model.state_dict())
        # if len(text_clients)!=0:
        #     self.txtmodel.load_state_dict(txt_aggregated_model.state_dict())
        # else:
        #     self.txtmodel = copy.deepcopy(self.server.model)

        del global_dict
        torch.cuda.empty_cache()
        gc.collect()

    def aggloot(self):
        print(f"-------------LOOT distilation---------------")
        clients = []
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

        mm_clients = []
        text_clients = []
        image_clients = []        
        for i, station in enumerate(self.clients):
            client_path = f"./CARckpts/TmpMAVG1/{self.args.prefix}_{i}.pth"  
            client_model = torch.load(client_path, map_location='cpu')

            if station.modality == "mm":
                mm_clients.append(client_model)
            elif station.modality == "text":
                text_clients.append(client_model)
            elif station.modality == "image":
                image_clients.append(client_model)

        mm_weight = 1 / len(mm_clients) if mm_clients else 0
        img_weight = 1 / len(image_clients) if image_clients else 0
        txt_weight = 1 / len(text_clients) if text_clients else 0

        # Aggregate models based on modality
        mm_aggregated_model = self.aggregate_models(mm_clients, mm_weight)
        img_aggregated_model = self.aggregate_models(image_clients, img_weight)
        txt_aggregated_model = self.aggregate_models(text_clients, txt_weight)

        self.mmmodel.load_state_dict(mm_aggregated_model.state_dict()) 
        self.imgmodel.load_state_dict(img_aggregated_model.state_dict())
        self.txtmodel.load_state_dict(txt_aggregated_model.state_dict())

        del mm_clients, image_clients, text_clients

        # Append aggregated models to clients list if they exist
        if mm_aggregated_model is not None:
            clients.append(mm_aggregated_model)
        if img_aggregated_model is not None:
            clients.append(img_aggregated_model)
        if txt_aggregated_model is not None:
            clients.append(txt_aggregated_model)            
        
        # for i, station in enumerate(self.clients):
        #     client_path = f"./CARckpts/TmpMAVG1/{self.args.prefix}_{i}.pth"  
        #     client_model = torch.load(client_path, map_location='cpu')
        #     clients.append(client_model)

        optimizers = [torch.optim.Adam(filter(lambda p: p.requires_grad, c.parameters()) , lr=1e-4) for c in clients]

       
        epoch_losses = [0.0] * len(clients)
        n_batches = 0
        cnt = 0
        with tqdm(self.server.val_loader, unit="batch") as tepoch:
            for data in tepoch:
                images = data["image"].cuda()
                texts = data["text"]
                feats = []
                for client in clients:
                    client.cuda()
                    client.eval()
                    out = client(images, texts, "", "mm")
                    feats.append((out['imagefeat'], out['textfeat']))
                n = len(clients)
                total_loss = torch.tensor(0.0).cuda()
                for i, (img_i, txt_i) in enumerate(feats):
                    loss_i = torch.tensor(0.0).cuda()
                    for j in range(n):
                        if j == i:
                            continue
                        img_j, txt_j = feats[j]
                        sim_img = F.cosine_similarity(img_i, img_j, dim=-1).mean()
                        sim_txt = F.cosine_similarity(txt_i, txt_j, dim=-1).mean()
                        loss_i = loss_i + -0.5 * (sim_img + sim_txt)
                    loss_i = loss_i / (n - 1)
                    epoch_losses[i] += loss_i.item()
                    total_loss = total_loss + loss_i

                for opt in optimizers:
                    opt.zero_grad()
                total_loss.backward()
                for opt in optimizers:
                    opt.step()

                n_batches += 1
                cnt +=1
                if cnt > 100:
                    break
        avg_losses = [l / cnt for l in epoch_losses]
        for idx, loss in enumerate(avg_losses):
            print(f"Client {idx} distill loss: {loss:.4f}")

        final_aggregated_model = self.aggregate_models(clients, 1 / len(clients))
        if final_aggregated_model is not None:
            self.server.model.load_state_dict(final_aggregated_model.state_dict())

        torch.cuda.empty_cache()
        gc.collect()      

    def aggdis(self):
        clients = []
        logits_list = []
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        
        for i, station in enumerate(self.clients):
            client_path = f"./CARckpts/TmpMAVG1/{self.args.prefix}_{i}.pth"  
            client_model = torch.load(client_path, map_location='cpu')
            clients.append(client_model)

        optimizers = [torch.optim.Adam(filter(lambda p: p.requires_grad, c.parameters()) , lr=1e-4) for c in clients]

        print(f"len clients   {len(clients)}")
        epoch_losses = [0.0] * len(clients)
        n_batches = 0
        cnt = 0
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        n = len(clients)
        cnt = 0
        tloss = 0.0

        with tqdm(self.server.val_loader, unit="batch") as tepoch:
            for data in tepoch:
                images = data["image"].cuda()
                texts = data["text"]

                # —— 修正1：每个 batch 重新初始化 —— 
                logits_list = []
                total_kl_loss = torch.tensor(0.0, device=images.device)

                # 1) forward 并收集 logits
                for client in clients:
                    client.cuda().eval()
                    out = client(images, texts, None, None, None, "mm")
                    logits_list.append(out['logits'])  # [B, C]

                # 2) 两两计算 KL 散度
                for i, logits_i in enumerate(logits_list):
                    # —— 修正2：使用 log 概率 —— 
                    log_p_i = F.logsigmoid(logits_i)    # log(sigmoid)
                    kl_i = torch.tensor(0.0, device=images.device)

                    for j, logits_j in enumerate(logits_list):
                        if j == i:
                            continue
                        q_j = torch.sigmoid(logits_j).detach()
                        kl_i = kl_i + kl_loss_fn(log_p_i, q_j)

                    total_kl_loss = total_kl_loss + kl_i / (n - 1)

                tloss += total_kl_loss.item()

                for opt in optimizers:
                    opt.zero_grad()
                total_kl_loss.backward()
                for opt in optimizers:
                    opt.step()

                cnt += 1
                if cnt > 100:
                    break

        avg_loss = 1.0 * tloss / cnt
        print(f"distill loss: {avg_loss:.4f}")
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
                if self.args.method == "momke":
                    self.clients[i].run1(comms, self.idxs, self.server.model)
                else:
                    self.clients[i].run(comms, self.img_vec, self.txt_vec, self.labels, self.idxs, self.server.model, self.mmmodel, self.imgmodel, self.txtmodel )
                
                tmploss += self.clients[i].loss                    
                
            tmploss /= 1.0*len(self.clients)
            print(f"avgloss {tmploss:.4f}")
            self.logger.info(f"avgloss {tmploss:.4f}")
            print(f"-----------Performing global aggregration--------------------------")
           
            # self.aggloot()
            self.aggregateCAR()

            print("---------------Evaluating Aggregrated Model in Val Set-------------------------------")
            if self.round > 15:
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
                    if self.args.method == "rapcc" or self.args.method == "ragpt":
                        output = self.server.model(frames, text, None, None, None, "ret")
                    elif self.args.method == "CAR":
                        output = self.server.model(frames, text, "", self.modality)
                    elif self.args.method == "momke":
                        output = self.server.model(frames, text, False, self.modality)

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
        
        # self.mappings = self.generate_RAG_mapping(self.server.val_loader, self.img_vec, labels, self.server.model, self.round)
        # self.mappings1 = self.generate_RAG_mapping(self.server.test_loader, self.img_vec, labels, self.server.model, self.round)
        # self.mappingsRSNA = self.generate_RAG_mapping(self.rsnadl, self.img_vec, labels, self.server.model, self.round)
        # self.mappingscxpneu = self.generate_RAG_mapping(self.cxpneudl, self.img_vec, labels, self.server.model, self.round)
        # torch.save(self.mappingsRSNA, f"./CARckpts/gptf/RSNA2Map.pth")

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

        # print(f"Length of local features: {len(local_img_vec)}")
        # print(f"Length of global features: {len(global_vec)}")
        # print(f"Max index from closest_indices: {closet_idx.max()}")
        mapping = {a:b for (a,b) in zip(local_idx, closet_idx)}
        model.cpu()
        return mapping


    def retrive_data(self, istest):
        # if istest == "val":
        #     map_idx = operator.itemgetter(*local_idxs)(self.mappings)
        # else:
        #     map_idx = operator.itemgetter(*local_idxs)(self.mappings1)
        # global_idx = operator.itemgetter(*map_idx)(self.idxs)
        
        global_idx = [0]
        # if self.round == 0:
        #     map_idx = operator.itemgetter(*local_idxs)(self.mappings1)
        #     global_idx = operator.itemgetter(*map_idx)(global_idxs)
        #     if self.cnt not in self.rdata:
        #         self.rdata[self.cnt] = {}
        #     self.rdata[self.cnt] = global_idx

        # else:
            # global_idx = self.rdata[self.cnt]

        global_idx = self.rdata[self.cnt]
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

# self, local_idxs, global_idxs, istest
    def retrive_datarsna(self):
        global_idx = [0]
        # if istest == "val":
        #     map_idx = operator.itemgetter(*local_idxs)(self.mappings)
        # else:
        #     map_idx = operator.itemgetter(*local_idxs)(self.mappings1)
        # global_idx = operator.itemgetter(*map_idx)(self.idxs)
        
        
        # if self.round == 0:
        #     map_idx = operator.itemgetter(*local_idxs)(self.mappingscxpneu)
        #     global_idx = operator.itemgetter(*map_idx)(global_idxs)
        #     if self.cnt not in self.rdata:
        #         self.rdatapneu[self.cnt] = {}
        #     self.rdatapneu[self.cnt] = global_idx

        # else:
        #     global_idx = self.rdatapneu[self.cnt]

        # torch.save(self.rdatapneu, f"./CARckpts/gptf/cxpneuridx.pth")

        global_idx = self.rdataRSNA[self.cnt]
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

  
