import torch
import torch.nn as nn
from typing import Dict, Any
from src.attacks.attacks import FGSM


class AdversarialTraining:
    """Adversarial training defense"""
    
    def __init__(self, attack_type='fgsm', attack_params=None):
        self.attack_type = attack_type
        self.attack_params = attack_params or {}
        self.attack = None
    
    def setup_attack(self, model, loss_fn, optimizer, input_shape, nb_classes):
        """Setup the attack for adversarial training"""
        if self.attack_type == 'fgsm':
            self.attack = FGSM(model, loss_fn, optimizer, input_shape, nb_classes, **self.attack_params)
        else:
            raise ValueError(f"Unsupported attack type: {self.attack_type}")
    
    def generate_adversarial_batch(self, x, y):
        """Generate adversarial examples for a batch"""
        if self.attack is None:
            raise RuntimeError("Attack not setup. Call setup_attack() first.")
        
        return self.attack.generate(x, y)
    
    def training_step(self, model, x, y, criterion, alpha=0.5):
        """
        Adversarial training step
        alpha: weight for adversarial loss (0.5 means equal weight for clean and adversarial)
        """
        # Clean loss
        clean_logits = model(x)
        clean_loss = criterion(clean_logits, y)
        
        # Generate adversarial examples
        x_adv = self.generate_adversarial_batch(x, y)
        
        # Adversarial loss
        adv_logits = model(x_adv)
        adv_loss = criterion(adv_logits, y)
        
        # Combined loss
        total_loss = (1 - alpha) * clean_loss + alpha * adv_loss
        
        return total_loss, clean_loss, adv_loss