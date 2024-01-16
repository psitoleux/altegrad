from dataloader import GraphTextDataset, GraphDataset, TextDataset
from dataloader_2 import LabelDataset
from info_nce import InfoNCE
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
import time
import os
import pandas as pd
import gc

torch.cuda.empty_cache()
gc.collect()

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
def InfoNCE_loss(v1,v2):
    return INCE(v1,v2)+INCE(v2,v1)


model_name = 'allenai/scibert_scivocab_uncased'; nout = 768 # scibert
#model_name =  'WhereIsAI/UAE-Large-V1'; nout = 1024 # UAE-Large
#model_name = 'llmrails/ember-v1'; nout = 1024 # ember

tokenizer = AutoTokenizer.from_pretrained(model_name)

gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]

#val_dataset = LabelDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
#train_dataset = LabelDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nb_epochs = 12
target_batch_size, batch_size = 32, 16 # target_batch_size : effective batch after accumulation steps
accumulation_steps = target_batch_size // batch_size
learning_rate = 2e-5
val_stop = 2


val_loader = DataLoader(val_dataset, batch_size=batch_size # num_workers = 4 + pin_memory = True supposed to speed up things
                        , shuffle=True, num_workers = 4, pin_memory=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size
                          , shuffle=True, num_workers = 4, pin_memory=True)



model = Model(model_name=model_name, num_node_features=300, nout=nout, nhid=300, graph_hidden_channels=300) # nout = model hidden dim
model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of trainable parameters in the model: ', total_params) 

scaler = torch.cuda.amp.GradScaler() #scaler : needed for AMP training
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=learning_rate*5,total_steps=nb_epochs* len(train_loader))

loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000


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



for i in range(epoch+1, epoch+nb_epochs+1):
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
        #y = batch.y#torch.FloatTensor(batch.y)
        #batch.pop('y')

        graph_batch = batch


        with torch.cuda.amp.autocast(): # mixed precision 
            x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
            current_loss = InfoNCE_loss(x_graph, x_text)
            #current_loss, pred = negative_sampling_contrastive_loss(x_graph, x_text,y.float())   

        scaler.scale(current_loss).backward()
        loss += current_loss.item()

        count_iter += 1
        

        if count_iter % accumulation_steps == 0:
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()

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
        #y = batch.y #torch.FloatTensor(batch.y)
        #batch.pop('y')

        graph_batch = batch
        with torch.no_grad():
            x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
            current_loss = InfoNCE_loss(x_graph, x_text)
            
            #current_loss, pred = negative_sampling_contrastive_loss(x_graph, x_text, y.float())   
            val_loss += current_loss.item()
    best_validation_loss = min(best_validation_loss, val_loss)

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
        if val_stop-j > 1:
            print('validation loss has not improved, ', val_stop - j, ' epochs before early stopping')
        elif val_stop - j == 1:
                print('validation loss has not improved, one epoch before early stopping')
        elif j == val_stop:
            if j == val_stop: # if val loss doesn't improve after val_stop epochs, stop training
                print('validation loss has not improved in ', val_stop, ' epoch(s), we stop training')
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
