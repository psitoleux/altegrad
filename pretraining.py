from dataloader import GraphTextDataset, GraphDataset, TextDataset
from loss import info_nce_loss
from info_nce import InfoNCE
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model, GATEncoder
import numpy as np

import torch
from torch import optim
import time
import os

import gc
from parsers import get_pretraining_parser

from tqdm import tqdm, trange

def pretraining_loss(v):
    return InfoNCE()(v,v)


args = get_pretraining_parser()

print(args)

nb_epochs = args.epochs
val_every = args.val_frequency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = args.lr
batch_size = args.batch_size

best_validation_loss = 1_000_000

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]

val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=len(val_loader), shuffle=True, num_workers=4, pin_memory=True)

nout = 768
num_node_features = args.num_node_features
nhid = args.nhid
graph_hidden_channels = args.graph_hidden_channels


graph_encoder = GATEncoder(num_node_features, nout, nhid, graph_hidden_channels).to(device)

total_params = sum(p.numel() for p in graph_encoder.parameters() if p.requires_grad)
print('Number of trainable parameters in the model: ', total_params) 


scaler = torch.cuda.amp.GradScaler()
optimizer = optim.AdamW(graph_encoder.parameters(), lr=lr,
                            betas=(0.9, 0.999),
                            weight_decay=args.weight_decay,
                        amsgrad = True)

total_steps = nb_epochs * len(train_loader)

scheduler_name = args.scheduler.lower()


if scheduler_name == 'reduce_on_plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, threshold=1e-4, threshold_mode='rel')
elif scheduler_name == 'one_cycle':
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr*2,total_steps=nb_epochs* len(train_loader))
elif scheduler_name == 'cosine_warmup':
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_epochs*total_steps // nb_epochs ,  num_training_steps = total_steps)
elif scheduler_name == 'cosine_warmup_restarts':
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_epochs*total_steps // nb_epochs ,  num_training_steps = total_steps, num_cycles = args.nb_cycles)



#nb_epochs_per_cycle = nb_epochs // args.nb_cycles
#step_size_up = len(train_loader) * nb_epochs_per_cycle

#scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr / 100, max_lr=lr, step_size_up=step_size_up, mode='triangular2'
#                                        , cycle_momentum = False)

save_path_ge = os.path.join('./pretrained/', 'graph_encoder.pt')


loss = 0

k = 0 
patience = 10

print('Pretraining graph encoder')
for i in range(nb_epochs):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    for j,batch in enumerate(train_loader):
        batch.pop('input_ids')
        batch.pop('attention_mask')

        with torch.cuda.amp.autocast():
            x_graph = graph_encoder(batch)
            current_loss = pretraining_loss(x_graph)

        scaler.scale(current_loss).backward()
        loss += current_loss.item()

        scaler.step(optimizer)
        optimizer.zero_grad(set_to_none=True)
        scaler.update()

        scheduler.step()

    print("Epoch ", i+1, "training loss: ", loss / (len(train_loader) / len(val_loader) ) )
    loss = 0
    

    if i % val_every == 0:

        val_loss = 0
        graph_encoder.eval()
        for batch in val_loader:
            batch.pop('input_ids')
            batch.pop('attention_mask')

            graph_batch = batch
            with torch.no_grad():
                x_graph = graph_encoder(batch.to(device))
                current_loss = pretraining_loss(x_graph)
        
            val_loss += current_loss.item()
    
        best_validation_loss = min(best_validation_loss, val_loss)
        print('validation loss: ', val_loss)
        if  best_validation_loss==val_loss:
            print('validation loss improved saving checkpoint...')
            
            dir_name = './pretrained/'
            files = os.listdir(dir_name)
            for item in files:
                if item.endswith(".pt"):
                    os.remove(os.path.join(dir_name, item))

            torch.save({'graph_encoder_state_dict': graph_encoder.state_dict(),}, save_path_ge)
            
            print('checkpoint saved to: {}'.format(save_path_ge))

            k = 0
        else:
            k += 1
        if k == patience:
            break
        
        scheduler.step(val_loss)
        

