from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import hydra
from omegaconf import DictConfig
from attacks.attacks import FGSM
from defenses.defenses import AdversarialTraining

class ResNet18Classifier(LightningModule):
    def __init__(self, num_classes: int, cfg: DictConfig = None):
        super().__init__()
        self.cfg = cfg
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.num_classes = num_classes
        
        # Instantiate criterion if provided
        if cfg and hasattr(cfg, 'criterion'):
            self.criterion = hydra.utils.instantiate(cfg.criterion)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Adversarial components
        self.adversarial_defense = None
        self.fgsm_attack = None
        self.adversarial_enabled = False

    def setup_adversarial(self, adversarial_cfg):
        """Setup adversarial training/testing components"""
        self.adversarial_enabled = True
        
        # Setup adversarial defense for training
        if hasattr(adversarial_cfg, 'defense') and adversarial_cfg.defense.enabled:
            self.adversarial_defense = AdversarialTraining(
                attack_type=adversarial_cfg.defense.attack_type,
                attack_params=adversarial_cfg.defense.attack_params
            )
        
        # Setup FGSM for evaluation
        if hasattr(adversarial_cfg, 'evaluation') and adversarial_cfg.evaluation.fgsm.enabled:
            # We'll setup FGSM attack in on_validation_epoch_start
            self.fgsm_params = adversarial_cfg.evaluation.fgsm
    
    def on_validation_epoch_start(self):
        """Setup FGSM attack for evaluation"""
        if self.adversarial_enabled and hasattr(self, 'fgsm_params'):
            # Get a dummy optimizer for ART (not used in evaluation)
            dummy_optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
            
            # Determine input shape based on your data
            # This assumes images - adjust based on your actual data
            input_shape = (3, 224, 224)  # Adjust based on your input
            
            self.fgsm_attack = FGSM(
                model=self.model,
                loss_fn=self.criterion,
                optimizer=dummy_optimizer,
                input_shape=input_shape,
                nb_classes=self.num_classes,
                eps=self.fgsm_params.get('eps', 0.3),
                clip_values=(0, 1)
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Standard training or adversarial training
        if self.adversarial_defense is not None:
            # Setup attack if not already done
            if self.adversarial_defense.attack is None:
                dummy_optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
                input_shape = x.shape[1:]  # Remove batch dimension
                self.adversarial_defense.setup_attack(
                    self.model, self.criterion, dummy_optimizer, 
                    input_shape, self.num_classes
                )
            
            # Adversarial training step
            total_loss, clean_loss, adv_loss = self.adversarial_defense.training_step(
                self.model, x, y, self.criterion, alpha=0.5
            )
            
            self.log("train_loss", total_loss, prog_bar=True)
            self.log("train_clean_loss", clean_loss)
            self.log("train_adv_loss", adv_loss)
            return total_loss
        else:
            # Standard training
            logits = self(x)
            loss = self.criterion(logits, y)
            self.log("train_loss", loss, prog_bar=True)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Clean evaluation
        clean_logits = self(x)
        clean_loss = self.criterion(clean_logits, y)
        clean_preds = torch.argmax(clean_logits, dim=1)
        clean_acc = torch.sum(clean_preds == y).float() / len(y)
        
        self.log("val_loss", clean_loss, prog_bar=True)
        self.log("val_clean_acc", clean_acc, prog_bar=True)
        
        # Adversarial evaluation if FGSM is available
        if self.fgsm_attack is not None:
            try:
                # Generate adversarial examples
                x_adv = self.fgsm_attack.generate(x, y)
                
                # Evaluate on adversarial examples
                adv_logits = self(x_adv)
                adv_loss = self.criterion(adv_logits, y)
                adv_preds = torch.argmax(adv_logits, dim=1)
                adv_acc = torch.sum(adv_preds == y).float() / len(y)
                
                self.log("val_adv_loss", adv_loss)
                self.log("val_fgsm_acc", adv_acc, prog_bar=True)
                
                # Log robustness metrics
                robustness = adv_acc / clean_acc if clean_acc > 0 else torch.tensor(0.0)
                self.log("val_robustness_ratio", robustness)
                
            except Exception as e:
                print(f"FGSM evaluation failed: {e}")
        
        return clean_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        # Clean evaluation
        clean_logits = self(x)
        clean_loss = self.criterion(clean_logits, y)
        clean_preds = torch.argmax(clean_logits, dim=1)
        clean_acc = torch.sum(clean_preds == y).float() / len(y)
        
        self.log("test_clean_loss", clean_loss)
        self.log("test_clean_acc", clean_acc)
        
        # Adversarial evaluation
        if self.fgsm_attack is not None:
            try:
                x_adv = self.fgsm_attack.generate(x, y)
                adv_logits = self(x_adv)
                adv_loss = self.criterion(adv_logits, y)
                adv_preds = torch.argmax(adv_logits, dim=1)
                adv_acc = torch.sum(adv_preds == y).float() / len(y)
                
                self.log("test_fgsm_loss", adv_loss)
                self.log("test_fgsm_acc", adv_acc)
                
                # Additional metrics
                robustness = adv_acc / clean_acc if clean_acc > 0 else torch.tensor(0.0)
                self.log("test_robustness_ratio", robustness)
                
            except Exception as e:
                print(f"FGSM test evaluation failed: {e}")
        
        return clean_loss

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