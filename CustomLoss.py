import torch
import torch.nn             as nn
import torch.nn.functional  as functional
from CustomLayers import ClosedForm

class CFLoss(nn.Module):
    def __init__(self, model, loss_function: callable, strength: float, features: int):
        super().__init__()
        self.loss_fn = loss_function
        self.strength = strength
        self.model = model
        self.closed_form = ClosedForm(features = features)

    def forward(self, input):
        l1_reg = 0
        for param in self.closed_form.parameters():
            # print(param)
            l1_reg += torch.linalg.vector_norm(param, 0)

        return self.strength * (self.loss_fn(self.model(input), self.closed_form(input))) + l1_reg