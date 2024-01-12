import os 
import os.path as osp
import torch
from torch_geometric.data import Dataset 
from torch_geometric.data import Data
from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import numpy as np
from dataloader import GraphTextDataset


class LabelDataset(GraphTextDataset):

    def __init__(self, root, gt, split, tokenizer=None, transform=None, pre_transform=None):
        # Weird error test
        super(LabelDataset, self).__init__(root, gt, split, tokenizer, transform, pre_transform)
        
    def process(self):
        i = 0        
        for raw_path in self.raw_paths:
            cid = int(raw_path.split('/')[-1][:-6])                
            
            text_input = self.tokenizer([self.description[1][cid]],
                                   return_tensors="pt", 
                                   truncation=True, 
                                   max_length=256,
                                   padding="max_length",
                                   add_special_tokens=True,)
            edge_index, x = self.process_graph(raw_path)
        

            data = Data(x=x, y = 1, edge_index=edge_index, input_ids=text_input['input_ids'], attention_mask=text_input['attention_mask'])

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1
            
            k = np.random.randint(len(self.raw_paths))
            while cid == self.raw_paths[k]:
                k = np.random.randint(len(self.raw_paths))
            
            raw_path_desc = self.raw_paths[k]
            cid_desc = int(raw_path_desc.split('/')[-1][:-6])
            
            text_input = self.tokenizer([self.description[1][cid_desc]],
                       return_tensors="pt", 
                       truncation=True, 
                       max_length=256,
                       padding="max_length",
                       add_special_tokens=True,)
        
            data = Data(x=x, y = 0, edge_index=edge_index, input_ids=text_input['input_ids'], attention_mask=text_input['attention_mask'])

            torch.save(data, osp.join(self.processed_dir, 'data_false{}.pt'.format(cid)))
            
            i += 1

