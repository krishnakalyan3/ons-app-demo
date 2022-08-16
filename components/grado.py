from lightning.app.components.serve import ServeGradio
import pytorch_lightning as pl

import torch.nn as nn
from torchvision import transforms as T

import timm
import wandb
import gradio as gr
from pathlib import Path
import os


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
        x = self.model(x)
        return x

class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CustomEffNet()
    
    def forward(self, x):
        x = self.model(x)
        return x

class ImageServeGradio(ServeGradio):

    inputs = gr.inputs.Image(type="pil", shape=(224, 224))
    outputs = gr.outputs.Label(num_top_classes=10)

    def __init__(self, cloud_compute, parallel=True, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, parallel=parallel, **kwargs)
        self.examples = None
        self.best_model_path = None
        self._transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self._labels = {0: "NORMAL", 1:"GLAUCOMA"}

    def run(self):
        self.examples = [os.path.join(str("./images"), f) for f in os.listdir("./images")]
        self._transform = self._transform
        super().run()

    def predict(self, img):
        img = self._transform(img)
        img = img.unsqueeze(0)
        prediction = self.model(img).argmax().item()
        return self._labels[prediction]

    def build_model(self):
        run = wandb.init(project="ons")
        artifact = run.use_artifact("krishnakalyan/ons/model-s2gdu5cx:v0", type="model")
        artifact_dir = artifact.download()
        model = ImageClassifier.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        return model
