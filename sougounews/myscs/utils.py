import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred, target):
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smoothed_labels = one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smoothed_labels * log_prob).sum(dim=1).mean()
        return loss

def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            progress_tensor = torch.tensor(progress)  # 转换为 Tensor
            return 0.5 * (1 + torch.cos(torch.pi * progress_tensor))  # 余弦退火
    return LambdaLR(optimizer, lr_lambda)