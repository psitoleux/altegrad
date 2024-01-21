from dataloader import GraphTextDataset, GraphDataset, TextDataset

from loss import contrastive_loss, negative_sampling_contrastive_loss, info_nce_loss
from info_nce import InfoNCE

from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model, GATEncoder
import numpy as np
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup
import torch
from torch import optim
import time
import os
import pandas as pd
import gc
from parsers import get_main_parser

from tqdm import tqdm, trange

args= get_main_parser()



model_name = args.model_name
nout = args.nout

tokenizer = AutoTokenizer.from_pretrained(model_name)

gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]

val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = args.epochs
target_batch_size, batch_size = args.target_batch_size, args.batch_size# target_batch_size : effective batch after accumulation steps
accumulation_steps = target_batch_size // batch_size
learning_rate = args.lr
patience = args.patience


val_loader = DataLoader(val_dataset, batch_size=batch_size # num_workers = 4 + pin_memory = True supposed to speed up things
                        , shuffle=True, num_workers = 4, pin_memory=True).to(device)
train_loader = DataLoader(train_dataset, batch_size=batch_size
                          , shuffle=True, num_workers = 4, pin_memory=True).to(device)


num_node_features, nhid, graph_hidden_channels = args.num_node_features, args.nhid, args.graph_hidden_channels
trainable = args.trainable

print(trainable)

model = Model(model_name=model_name, num_node_features=num_node_features
              , nout=nout, nhid=nhid, graph_hidden_channels=graph_hidden_channels,
              trainable=trainable) # nout = model hidden dim
model.to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of trainable parameters in the model: ', total_params) 

scaler = torch.cuda.amp.GradScaler() #scaler : needed for AMP training

optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01, amsgrad=True)

scheduler_name = args.scheduler.lower()

total_steps = nb_epochs * len(train_loader)

print(scheduler_name)

if scheduler_name == 'reduce_on_plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, threshold=1e-4, threshold_mode='rel')
elif scheduler_name == 'one_cycle':
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,max_lr=learning_rate*2,total_steps=nb_epochs* len(train_loader))
elif scheduler_name == 'cosine_warmup':
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_epochs*total_steps // nb_epochs ,  num_training_steps = total_steps)
elif scheduler_name == 'cosine_warmup_restarts':
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_epochs*total_steps // nb_epochs ,  num_training_steps = total_steps, num_cycles = args.nb_cycles)




dir_name = './'
files = os.listdir(dir_name)
chkpt = []
for item in files:
    if item.endswith(".pt"):
        chkpt += [os.path.join(dir_name, item)]
chkpt = sorted(chkpt)


if len(chkpt) != 0:
  print('loading checkpoint...')
  checkpoint = torch.load(chkpt[-1])
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']  
  print('Done!')


loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = args.print_every
best_validation_loss = 1000000
epoch = 0



pretrained_graph_encoder = args.pretrained_graph_encoder


if pretrained_graph_encoder is not None:
    
    print('Loading pretrained graph encoder...')
    model.load_pretrained_graph_encoder(pretrained_graph_encoder)
    print('Done!')



for i in range(epoch, epoch+nb_epochs):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory=True)
    print('-----EPOCH{}-----'.format(i+1))
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for batch in train_loader:
        torch.cuda.empty_cache()
        gc.collect()
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')

        graph_batch = batch


        with torch.cuda.amp.autocast(): # mixed precision 
            x_graph, x_text = model(graph_batch, 
                                input_ids, 
                                attention_mask)
            current_loss = info_nce_loss(x_graph, x_text) 

        scaler.scale(current_loss).backward()
        loss += current_loss.item()

        count_iter += 1
        

        scaler.step(optimizer)
        optimizer.zero_grad(set_to_none=True)
        scaler.update()

        scheduler.step()
        
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0 

    val_loss = 0
    model.eval()
    for batch in val_loader:
        torch.cuda.empty_cache()
        gc.collect()
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')

        graph_batch = batch
        with torch.no_grad():
            x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
            current_loss = info_nce_loss(x_graph, x_text)
            
            val_loss += current_loss.item()
    best_validation_loss = min(best_validation_loss, val_loss)
    
    #scheduler.step(val_loss) #for reduce LR on plateau
    
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )
    if best_validation_loss==val_loss:
        print('validation loss improved saving checkpoint...')
        save_path = os.path.join('./', 'model'+str(i)+'.pt')

        dir_name = './'
        files = os.listdir(dir_name)
        for item in files:
            if item.endswith(".pt"):
                os.remove(os.path.join(dir_name, item))

        torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_accuracy': val_loss,
        'loss': loss,
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))
        j = 0
    else:
        j += 1
        
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if patience-j > 1:
            print('validation loss has not improved, ', patience - j, ' epochs before early stopping')
        elif patience - j == 1:
                print('validation loss has not improved, one epoch before early stopping')
        elif j == patience:
            if j == patience: # if val loss doesn't improve after patience epochs, stop training
                print('validation loss has not improved in ', patience, ' epoch(s), we stop training')
                break


torch.cuda.empty_cache()
gc.collect()
print('loading best model...')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


with torch.no_grad():

    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()

    test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

    idx_to_cid = test_cids_dataset.get_idx_to_cid()

    test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)


    print('Creating test graph embeddings...')
    graph_embeddings = []
    for batch in test_loader:
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())

    test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
    print('Creating test text embeddings...')
    text_embeddings = []
    for batch in test_text_loader:
        for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())


from sklearn.metrics.pairwise import cosine_similarity


print('Computing the similarity between embeddings...')
similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]

print('Saving results...')
solution.to_csv('submission.csv', index=False)
print('Done!')
