from lightning import LightningDataModule
from torchvision import transforms
from torchvision.datasets import PCAM
from torch.utils.data import DataLoader

from src.utils.utils import train_transform, val_transform

class PCAMDataModule(LightningDataModule):
    def __init__(self, data_root: str, cfg=None, name=""):
        super().__init__()
        self.train = PCAM(root=data_root, split='train', transform=train_transform, download=True)

        self.val = PCAM(root=data_root, split='val', transform=val_transform, download=True)

        self.test = PCAM(root=data_root, split='test', transform=val_transform, download=True)
        
        self.cfg = cfg
        self.name = name

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.train
            self.val_dataset = self.val
        if stage == 'test' or stage is None:
            self.test_dataset = self.test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)
        else:
            return None

    def predict_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)
        else:
            return None
