import torch 
from info_nce import InfoNCE

CE = torch.nn.CrossEntropyLoss()
def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

BCEL = torch.nn.BCEWithLogitsLoss()
def negative_sampling_contrastive_loss(v1, v2, labels):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  eye = torch.diag_embed(labels).to(v1.device)
  return BCEL(logits, eye) + BCEL(torch.transpose(logits, 0, 1), eye), logits.diag() > 0


INCE = InfoNCE()
def info_nce_loss(v1,v2):
    return INCE(v1,v2)+INCE(v2,v1)
