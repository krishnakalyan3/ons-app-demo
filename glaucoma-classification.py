#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import torchmetrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import timm
import wandb
import os

from datetime import datetime
current_time = datetime.now().strftime("%D %H:%M:%S"); current_time


# In[ ]:


import s3fs
fs = s3fs.S3FileSystem(key=os.environ['DEMO_AWS_ACCESS_KEY_ID'], secret=os.environ['DEMO_AWS_SECRET_ACCESS_KEY'])
_ = fs.get("ons-classification", "data", recursive=True)


# In[2]:


seed_everything(1111)
wandb_logger = WandbLogger(project="ons", name=f"script-{current_time}", log_model="all")


# In[3]:


class ONS(Dataset):
    def __init__(self, 
                 data_dir: str, 
                 transforms = T.Compose([T.RandomCrop(256), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])):
        super().__init__()
        self.train_files = [i for i in data_dir.rglob("*.png")]
        self.transforms = transforms
        self.target_dict = {"NORMAL": 0, "GLAUCOMA": 1, "normal":0, "glaucoma":1}

    def __len__(self):
        return len(self.train_files)
    
    def __getitem__(self, idx):
        img_path  = self.train_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transforms:
            img = self.transforms(img)
        
        return {'image':img, 'target': self.target_dict[img_path.parent.name]}


# In[4]:


def get_transform(phase: str, size=224):
    if phase == 'train':
        return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# In[5]:


ROOT = Path("/content/data")
dataset_train = ONS(ROOT / "ons-test", get_transform("train"))
dataset_validation = ONS(ROOT / "ons-train", get_transform("val"))


# In[6]:


dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=32, shuffle=False, num_workers=4)


# In[7]:


wandb_logger.log_image(key="samples", images=[dataset_train[0]['image'], dataset_train[30]['image']], 
                       caption=[dataset_train[0]['target'], dataset_train[30]['target']])


# In[8]:


class CustomEffNet(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', num_classes=2, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# In[9]:


class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1Score()
        self.model = CustomEffNet()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = batch['image']
        y = batch['target']
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1(y_hat, y)
        
        logs = {'train_loss': loss, 'train_accuracy': acc, "train_f1": f1}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = batch['image']
        y = batch['target']
        y_hat = self.model(x)        
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1(y_hat, y)
        
        logs = {'valid_loss': loss, 'val_accuracy': acc, "val_f1": f1}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.03)


# In[10]:


checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")

trainer = pl.Trainer(max_epochs=7, 
                     accelerator='gpu',
                     devices=1, 
                     precision=16,
                     enable_progress_bar=True, 
                     callbacks=[checkpoint_callback],
                     logger=wandb_logger)    


# In[11]:


model = ImageClassifier()

trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_validation)


# In[12]:


wandb.finish()


# In[ ]:




