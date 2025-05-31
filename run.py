import os
import os.path as osp
import datetime
import torch
import torch.nn as nn

import lightning as l
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd

def train(cfg: DictConfig):
    l.seed_everything(cfg.seed, workers=True)

    tb_logger = TensorBoardLogger(save_dir=osp.expanduser(cfg.io.base_output_path), name=cfg.tag, version=cfg.io.version)

    # Pytorch Lightning module
    print("===> Start building model")
    model = hydra.utils.instantiate(cfg.model)
    print('\n\nMODEL INITIATED', model)
    
    # DataLoader - fix data_root path to use original working directory
    print("===> Loading datasets")
    original_cwd = get_original_cwd()
    cfg.data_module.data_root = osp.join(original_cwd, cfg.data_module.data_root.lstrip('./'))
    data_module = hydra.utils.instantiate(cfg.data_module)
   
    print("===> Instantiate callbacks")
    callbacks = [hydra.utils.instantiate(callback) for _, callback in cfg.callbacks.items()]

    # Trainer
    print("===> Instantiate trainer")
    if hasattr(cfg.trainer, 'strategy') and cfg.trainer.strategy == "ddp":
        trainer_cfg = dict(cfg.trainer)
        del trainer_cfg['strategy']
        trainer = Trainer(
            **trainer_cfg,
            logger=tb_logger,
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=getattr(cfg, 'find_unused_parameters', False), 
                               timeout=datetime.timedelta(seconds=365 * 24 * 3600))
        )
    else:
        trainer = Trainer(
            **cfg.trainer,
            logger=tb_logger,
            callbacks=callbacks
        )
        
    # Train the model
    if getattr(cfg, 'training', True):
        print("===> Start training")
        trainer.fit(model, data_module)

    if getattr(cfg, 'final_validate', False):
        print("===> Start validation")
        trainer.validate(model, data_module)
    if getattr(cfg, 'final_test', False):
        print("===> Start testing")
        trainer.test(model, data_module)

    if not getattr(cfg, 'training', True):
        checkpoint_dirpath = osp.join(cfg.io.base_output_path, cfg.tag, f"version_{cfg.io.version}")
        os.makedirs(checkpoint_dirpath, exist_ok=True)
        trainer.save_checkpoint(f"{checkpoint_dirpath}/last.ckpt")


@hydra.main(config_path="./config", config_name="default", version_base="1.1")
def run(cfg: DictConfig):
    print(f"PyTorch-Lightning Version: {l.__version__}")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = os.environ.get("HYDRA_FULL_ERROR", "1")
    run()