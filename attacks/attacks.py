from abc import ABC, abstractmethod
import torch
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import *


class BaseAttack(ABC):
    """Base class for adversarial attacks using ART"""
    
    def __init__(self, model, loss_fn, optimizer, input_shape, nb_classes, clip_values=(0, 1)):
        self.model = model
        self.classifier = PyTorchClassifier(
            model=model,
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=clip_values
        )
    
    @abstractmethod
    def generate(self, x, y=None, **kwargs):
        """Generate adversarial examples"""
        pass
    
    def _to_numpy(self, tensor):
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _to_tensor(self, array, device='cpu'):
        """Convert numpy array to tensor"""
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).to(device)
        return array

class FGSM(BaseAttack):
    """Fast Gradient Sign Method attack using ART"""
    
    def __init__(self, model, loss_fn, optimizer, input_shape, nb_classes, 
                 eps=0.3, eps_step=0.1, clip_values=(0, 1)):
        super().__init__(model, loss_fn, optimizer, input_shape, nb_classes, clip_values)
        self.attack = FastGradientMethod(
            estimator=self.classifier,
            eps=eps,
            eps_step=eps_step,
            targeted=False
        )
    
    def generate(self, x, y=None, **kwargs):
        """Generate FGSM adversarial examples"""
        x_np = self._to_numpy(x)
        y_np = self._to_numpy(y) if y is not None else None
        
        x_adv = self.attack.generate(x=x_np, y=y_np, **kwargs)
        
        device = x.device if hasattr(x, 'device') else 'cpu'
        return self._to_tensor(x_adv, device)