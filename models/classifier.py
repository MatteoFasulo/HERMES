from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import hydra
from omegaconf import DictConfig

class ResNet18Classifier(LightningModule):
    def __init__(self, num_classes: int, cfg: DictConfig = None):
        super().__init__()
        self.cfg = cfg
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Instantiate criterion if provided
        if cfg and hasattr(cfg, 'criterion'):
            self.criterion = hydra.utils.instantiate(cfg.criterion)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        if self.cfg and hasattr(self.cfg, 'optimizer'):
            optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=self.parameters())
        else:
            optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        
        if self.cfg and hasattr(self.cfg, 'scheduler'):
            scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer