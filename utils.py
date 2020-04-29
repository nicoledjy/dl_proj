import torch
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  
    return y[labels]            
    

def focal_loss(x, y):
    '''Focal loss.
    Args:
        x: (tensor) sized [N,D].
        y: (tensor) sized [N,].
    Return:
        (tensor) focal loss.
    '''
    alpha = 0.25
    gamma = 2
    num_classes = 9
    # turn labels into one-hot embeeddings
    t = one_hot_embedding(y, num_classes)  # [N,21]

    p = x.sigmoid()
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    
    return F.binary_cross_entropy_with_logits(x, t, w.detach(), reduction="sum")