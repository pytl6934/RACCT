import torch
from torch import nn
import torch.nn.functional as F
from .vilt import ViltModel
from .modules import MMG, CAP
from models.vilt.modeling_vilt import ViltModel
from models.vilt.image_processing_vilt import ViltImageProcessor
from transformers import BertTokenizer
from torchvision.transforms import Resize
import math
from PIL import Image
import copy
from peft import get_peft_model, LoraConfig, TaskType

image_processor = ViltImageProcessor.from_pretrained('./models/viltb32')
image_processor.do_rescale = False
tokenizer = BertTokenizer.from_pretrained('./models/viltb32', do_lower_case=True)
max_text_len = 128

previlt = ViltModel.from_pretrained('./models/viltb32')

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=2,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"]
)
previlt = get_peft_model(previlt, lora_config)  
print("--------------------lora-----------------------")

class Converter(nn.Module):
    def __init__(self, dim, embed_dim, is_Fusion=False):
        super().__init__()
        if is_Fusion:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        else:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        
        self.pool = nn.AdaptiveAvgPool1d(output_size=128)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.act(self.conv(x))
        x = self.pool(x)
        x = x.permute(0,2,1)
        return x

def denormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(image_tensor.device)
    return torch.clamp(image_tensor * std + mean, 0, 1)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class RACCT(torch.nn.Module):
    def __init__(self, 
                 prompt_position,
                 prompt_length,
                 dropout_rate,
                 lbd,
                 max_text_len = 128,
                 max_image_len = 145,                 
                 hs: int = 768):
        super(RACCT, self).__init__()

        self.max_text_len = max_text_len
        self.embedding_layer = previlt.embeddings
        self.encoder_layer = copy.deepcopy(previlt.encoder.layer)
        self.layernorm = copy.deepcopy(previlt.layernorm)
        self.pooler = copy.deepcopy(previlt.pooler)
        self.prompt_length = prompt_length
        self.prompt_position = prompt_position
        self.max_image_len = max_image_len
        self.hs = hs
        self.lbd=lbd

        cls_num = 14
        self.MMG = Converter(self.hs, self.hs)
        self.MMG_t = Converter(self.hs, self.hs)
        self.pool = nn.AdaptiveAvgPool1d(output_size=max_text_len)
        self.poolt = nn.AdaptiveAvgPool1d(output_size=max_text_len)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hs,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True    
        )
        self.cross_attn_t = nn.MultiheadAttention(
            embed_dim=self.hs,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True    
        )


        disease_emb = torch.load("disembed.pth")
        self.disease_aware_prompt = nn.Parameter(disease_emb)
        self.disease_aware_prompt.requires_grad = False

        self.classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 14),
            )

        self.classifier.apply(init_weights)

        for p in self.embedding_layer.parameters(): p.requires_grad = False
        for layer in self.encoder_layer:
            for p, value in layer.named_parameters(): 
                if "lora" in p :
                    value.requires_grad = True
                else:
                    value.requires_grad = False
        for p in self.layernorm.parameters(): p.requires_grad = False

    def forward(self, img, txt, rimg, rtxt, rl, modality):
        B = img.size(0)
        img = denormalize(img)
        
        text_encoding = tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )  
        input_ids = text_encoding['input_ids']
        attention_mask = text_encoding['attention_mask']
        token_type_ids = text_encoding['token_type_ids']
        input_ids = torch.tensor(input_ids,dtype=torch.int64).cuda()
        token_type_ids = torch.tensor(token_type_ids,dtype=torch.int64).cuda()
        attention_mask = torch.tensor(attention_mask,dtype=torch.int64).cuda()

        image_encoding = image_processor(img, return_tensors="pt")
        pixel_values = image_encoding["pixel_values"].cuda()
        pixel_mask = image_encoding["pixel_mask"].cuda()

        embedding, attn = self.embedding_layer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            image_token_type_idx=1,
            inputs_embeds=None,
            image_embeds=None            
        )

        text_emb = embedding[:, :self.max_text_len, :]
        image_emb = embedding[:, self.max_text_len:self.max_text_len+self.max_image_len, :]

        imagefeat = image_emb.mean(dim = 1)
        textfeat = text_emb.mean(dim = 1)    

        if modality == "ret":
            return {"imagefeat":imagefeat, "textfeat":textfeat} 

        #  ------------------------------------------

        if modality != 'mm':
            rimg = denormalize(rimg)
            rl=rl.float()
            text_encoding = tokenizer(
                rtxt,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_special_tokens_mask=True,
            )  
            input_ids = text_encoding['input_ids']
            attention_mask = text_encoding['attention_mask']
            token_type_ids = text_encoding['token_type_ids']
            input_ids = torch.tensor(input_ids,dtype=torch.int64).cuda()
            token_type_ids = torch.tensor(token_type_ids,dtype=torch.int64).cuda()
            attention_mask = torch.tensor(attention_mask,dtype=torch.int64).cuda()

            image_encoding = image_processor(rimg, return_tensors="pt")
            pixel_values = image_encoding["pixel_values"].cuda()
            pixel_mask = image_encoding["pixel_mask"].cuda()

            rembedding, attn = self.embedding_layer(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                image_token_type_idx=1,
                inputs_embeds=None,
                image_embeds=None            
            )

            rtext_emb = rembedding[:, :self.max_text_len, :]
            rimage_emb = rembedding[:, self.max_text_len:self.max_text_len+self.max_image_len, :]
            rtextfeat = rtext_emb.mean(dim = 1)    

            # if CAR-MFL setting 
            # image_emb = image_emb.permute(0,2,1)
            # image_emb = self.pool(image_emb)
            # image_emb = image_emb.permute(0,2,1) 
            # rimage_emb = rimage_emb.permute(0,2,1)
            # rimage_emb = self.poolt(rimage_emb)
            # rimage_emb = rimage_emb.permute(0,2,1) #else SKIP and comment this

        contextp_batch = self.disease_aware_prompt.unsqueeze(0)      
        contextp_batch = contextp_batch.expand(img.size(0), -1, -1)  # [B, 14, 768]

        if modality == "image":
            attn_img, attn_weights1 = self.cross_attn(
                query=image_emb,       # [B, Q, D]
                key=contextp_batch,        # [B, T, D]
                value=contextp_batch,      # [B, T, D]
                #  key_padding_mask=key_padding_mask
            )
            rectext_emb = self.MMG(attn_img)
            text_emb = self.lbd * rectext_emb + (1-self.lbd) * rtext_emb
            # text_emb =  rtext_emb

        # if modality == "text":
        #     attn_txt, attn_weights1 = self.cross_attn_t(
        #         query=text_emb,       # [B, Q, D]
        #         key=contextp_batch,        # [B, T, D]
        #         value=contextp_batch,      # [B, T, D]
        #         #  key_padding_mask=key_padding_mask
        #     )
        #     recimage_emb = self.MMG_t(attn_txt)
        #     image_emb = self.lbd * recimage_emb + (1-self.lbd) * rimage_emb

        output = torch.cat([text_emb, image_emb], dim=1)

        for i, layer_module in enumerate(self.encoder_layer):
            if i == self.prompt_position:
                N = output.shape[0]
                attention_mask = torch.cat([torch.ones(N, output.size(1)-attention_mask.size(1) ).cuda(), attention_mask], dim=1)
                layer_outputs = layer_module(output,"","",attention_mask=attention_mask)
                output = layer_outputs[0]
            else:
                layer_outputs = layer_module(output, "","",attention_mask=attention_mask)
                output = layer_outputs[0]
        output = self.layernorm(output)
        outputf = self.pooler(output)
        output = self.classifier(outputf)
        
        return {"logits": output, "imagefeat":imagefeat, "textfeat":textfeat, "feat": outputf}


