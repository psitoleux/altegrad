import torch
from dataloader import GraphDataset, TextDataset
from transformers import AutoTokenizer
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model

import os
import pandas as pd
import numpy as np

from parsers import get_inference_parser
from sklearn.metrics.pairwise import cosine_similarity

from tqdm.notebook import tqdm

args = get_inference_parser()


if args.save_path == '':
    dir_name = './'
    files = os.listdir(dir_name)
    chkpt = []
    for item in files:
        if item.endswith(".pt"):
            chkpt += [os.path.join(dir_name, item)]
    chkpt = sorted(chkpt)
    save_path = chkpt[-1]
else:
    save_path = args.save_path


model_name = 'allenai/scibert_scivocab_uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = args.batch_size



num_node_features, nhid, graph_hidden_channels = 300, 300, 300

model = Model(model_name=model_name, num_node_features=num_node_features
              , nout=nout, nhid=nhid, graph_hidden_channels=graph_hidden_channels) # nout = model hidden dim
model.to(device)


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
    for batch in tqdm(test_loader):
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())

    test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
    print('Creating test text embeddings...')
    text_embeddings = []
    for batch in tqdm(test_text_loader):
        for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())




print('Computing the similarity between embeddings...')
similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]

print('Saving results...')
solution.to_csv('submission.csv', index=False)
print('Done!')

