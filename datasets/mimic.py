import pickle
import re
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

idx_to_disease = {
    0: "Atelectasis",
    1: "Cardiomegaly",
    2: "Consolidation",
    3: "Edema",
    4: "Enlarged Cardiomediastinum",
    5: "Fracture",
    6: "Lung Lesion",
    7: "Lung Opacity",
    8: "No Finding",
    9: "Pleural Effusion",
    10: "Pleural Other",
    11: "Pneumonia",
    12: "Pneumothorax",
    13: "Support Devices"
}

def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ').replace('\r', '') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report



class MimicMultiModal(Dataset):
    def __init__(self, view_type="view1", split="train"):
        super().__init__()

        annFile = f"./annot/mimic-cxr-APPA.pkl"
        # annFile = f"./annot/tsnedata200-all.pkl"
        with open(annFile, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)
        self.data = loaded_data[split]
        if split=="train":
            self.transforms = transforms.Compose([
                    transforms.Resize((224, 224)),  # Resize the image to a fixed size (e.g., 224x224)
                    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally for data augmentation
                    transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
                    transforms.ToTensor(),  # Convert the image to a PyTorch tensor (0-1 range)
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image using precomputed values
                ])
        else:
            self.transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])   
        self.image_root = "./data/MIMICCXR/mimic-cxr-resized/files/"
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data_item = self.data[idx]
        report = data_item['report']

        # relative_path = data_item["image_path"][0].replace("jpg", "png")
        relative_path = data_item["image_path"][0]
        img_path = self.image_root + relative_path
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        label = data_item["label"]

        positive_indices = [i for i, v in enumerate(label) if v == 1]
        positive_diseases = [idx_to_disease[i] for i in positive_indices]
        if positive_diseases:
            prompt = "the photo shows sign of " + ", ".join(positive_diseases)
        else:
            prompt = "the photo shows no notable findings"

        cleaned_report = clean_report_mimic_cxr(report)
        return {"image":image, "label": torch.tensor(label), "text": cleaned_report, "pro":prompt, "idx": idx, "img_path": img_path}


class MimicPublic(Dataset):
    def __init__(self, dst_type="train"):
        super().__init__()

        annFile =f'./annot/mimic-cxr-APPA.pkl'
        with open(annFile, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)
        self.data = loaded_data["train"]
        if dst_type == "train":
            self.transforms = transforms.Compose([
                    transforms.Resize((224, 224)),  # Resize the image to a fixed size (e.g., 224x224)
                    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally for data augmentation
                    transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
                    transforms.ToTensor(),  # Convert the image to a PyTorch tensor (0-1 range)
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image using precomputed values
                ])
        else:
            self.transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])   
        self.image_root = "./data/MIMICCXR/mimic-cxr-resized/files/"

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data_item = self.data[idx]
        report = data_item['report']

        # relative_path = data_item["image_path"][0].replace("jpg", "png")
        relative_path = data_item["image_path"][0]
        img_path = self.image_root + relative_path
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        label = data_item["label"]
        positive_indices = [i for i, v in enumerate(label) if v == 1]
        positive_diseases = [idx_to_disease[i] for i in positive_indices]
        if positive_diseases:
            prompt = "the photo shows sign of " + ", ".join(positive_diseases)
        else:
            prompt = "the photo shows no notable findings"

        cleaned_report = clean_report_mimic_cxr(report)
        return {"image":image, "label": torch.tensor(label), 
        "text": cleaned_report, "pro":prompt, "idx": idx, "img_path": img_path}


class MimicMultiModal1(Dataset):
    def __init__(self, data_root, ann_root, view_type="view1", split="train"):
        super().__init__()

        annFile = f"./annot/mimic-cxr-APPA.pkl"
        with open(annFile, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)
        self.data = loaded_data[split]

        if split == "train":
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.image_root = "./data/MIMICCXR/mimic-cxr-resized/files/"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        report = data_item['report']
        relative_path = data_item["image_path"][0]
        img_path = self.image_root + relative_path

        # Load image and apply transformations
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)

        label = data_item["label"]
        cleaned_report = clean_report_mimic_cxr(report)

        missing_modality = random.random() < 0.2  # 20% chance of missing a modality
        missing_img = missing_modality and random.choice([True, False])  # If missing a modality, randomly choose which one

        if missing_img:
            image = None  # Mark image as missing
        else:
            cleaned_report = None  # Mark text as missing

        return image, torch.tensor(label), cleaned_report, idx




