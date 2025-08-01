import pickle
import re
import torch
import random
from PIL import Image
import pydicom
from torch.utils.data import Dataset
from torchvision import transforms
import os 
import csv

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



class RSNA2(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        self.lbl = "./data/RSNAdata/train_labels1.csv"
        self.image_root = "./data/RSNAdata/train_images"

        # Read CSV file and store data
        self.data = []
        with open(self.lbl, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.data.append(row)

        if split == "train":
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        patient_id = row['patientId']
        img_path = os.path.join(self.image_root, f"{patient_id}.dcm")

        # Read DICOM file
        dicom = pydicom.dcmread(img_path)
        image = Image.fromarray(dicom.pixel_array).convert('RGB')

        # Create label vector ulbl
        ulbl = [0] * 14
        target_value = int(eval(row['Target']))
        ulbl[7] = 0 if target_value == 0 else 1

        if self.transforms:
            image = self.transforms(image)

        return {"image": image, "text": "None" , "label":torch.tensor(ulbl), "idx": idx}



