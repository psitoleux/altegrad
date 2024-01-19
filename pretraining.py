from dataloader import GraphTextDataset, GraphDataset, TextDataset
from loss import info_nce_loss
from info_nce import InfoNCE
from transformers import AutoTokenizer
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

nb_epochs = args.epochs
val_every = args.val_frequency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = args.lr
batch_size = args.batch_size

pt_best_validation_loss = 1_000_000

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)


nout = 768
num_node_features, nhid, graph_hidden_channels = 300, 300, 300
graph_encoder = GATEncoder(num_node_features, nout, nhid, graph_hidden_channels).to(device)

optimizer_pt = optim.AdamW(graph_encoder.parameters(), lr=lr,
                            betas=(0.9, 0.999),
                            weight_decay=0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_pt, factor=0.7, patience=1, threshold=1e-4, threshold_mode='rel')


save_path_ge = os.path.join('./pretrained/', 'graph_encoder.pt')


loss = 0

val_loader = DataLoader(val_dataset, batch_size=batch_size_pt, shuffle=True, num_workers=4, pin_memory=True)

k = 0 

print('Pretraining graph encoder')
for i in range(nb_epochs_pt):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_pt, shuffle=True, num_workers=4, pin_memory=True)
    for j,batch in enumerate(train_loader):
        batch.pop('input_ids')
        batch.pop('attention_mask')

        with torch.cuda.amp.autocast():
            x_graph = graph_encoder(batch)
            current_loss = pretraining_loss(x_graph)

        scaler.scale(current_loss).backward()
        loss_pt += current_loss.item()

        scaler.step(optimizer)
        optimizer_pt.zero_grad(set_to_none=True)
        scaler.update()

        

    print("Epoch ", i+1, "training loss: ", loss_pt)
    loss_pt = 0
    

    if i % val_every == 0:

        val_loss = 0
        graph_encoder.eval()
        for batch in val_loader_pt:
            batch.pop('input_ids')
            batch.pop('attention_mask')

            graph_batch = batch
            with torch.no_grad():
                x_graph = graph_encoder(batch.to(device))
                current_loss = pretraining_loss(x_graph)
        
            val_loss += current_loss.item()
    
        best_validation_loss = min(pt_best_validation_loss, pt_val_loss)
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

        scheduler_pt.step(val_loss)
        

