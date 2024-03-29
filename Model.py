from torch import nn
import torch.nn.functional as F
import torch

from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel

import numpy as np

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()

        self.ln_hid = nn.LayerNorm((nhid))
        self.ln_out = nn.LayerNorm((nout))


        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid, bias = False)
        self.mol_hidden2 = nn.Linear(nhid, nout, bias = False)

    def mlp(self, x):
        x = self.ln_hid(self.mol_hidden1(x).relu())
        x = self.ln_out(self.mol_hidden2(x))

        return x

    def get_batch(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        return x, edge_index, batch



class GCNEncoder(GraphEncoder):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        GraphEncoder.__init__(self, num_node_features, nout, nhid, graph_hidden_channels)

        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)

    def forward(self, graph_batch):

        x, edge_index, batch = self.get_batch(graph_batch)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)

        x = self.mlp(x)
        
        return x


class GATEncoder(GraphEncoder):
    def __init__ (self, num_node_features, nout,  nhid, graph_hidden_channels,n_layers = 3):
        GraphEncoder.__init__(self,num_node_features, nout, nhid, graph_hidden_channels)
        
        self.layers = []
        self.layers += [GATv2Conv(num_node_features, graph_hidden_channels)]
        
        for i in range(n_layers-1):
            self.layers += [GATv2Conv(graph_hidden_channels, graph_hidden_channels)]
        
        self.layers = nn.ModuleList(self.layers)

    def forward(self, graph_batch): 

        x, edge_index, batch = self.get_batch(graph_batch)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = x.relu()

        x = global_mean_pool(x,batch)
        x = self.mlp(x)

        return x

class GTransEncoder(GraphEncoder):
    def __init__(self, num_node_features, nout,  nhid, graph_hidden_channels,n_layers=3):
        GraphEncoder.__init__(self,num_node_features, nout, nhid, graph_hidden_channels)

        num_heads = 4

        self.layers = []
        self.layers += [TransformerConv(num_node_features, graph_hidden_channels // num_heads, heads = num_heads)]

        for i in range(n_layers-1):
            self.layers += [TransformerConv(graph_hidden_channels, graph_hidden_channels // num_heads, heads=num_heads)]

        self.layers = nn.ModuleList(self.layers)

    def forward(self, graph_batch):
        x, edge_index, batch = self.get_batch(graph_batch)
        
        for layer in self.layers:
           x = layer(x, edge_index)
           x = x.relu()
        
        x = global_mean_pool(x,batch)
        x = self.mlp(x)

        return x




class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        
    def forward(self, input_ids, attention_mask): 
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:,0,:]

    def set_trainable_layers(self, trainable_layers):
        nparams_list = [(p.numel()) for p in self.bert.parameters() if p.requires_grad]

        if trainable_layers == 'all_but_embeddings':
            print('Not training embedding layer')
            for i,p in enumerate(self.bert.parameters()):
                if i in np.arange(len(nparams_list))[:6]:  
                    p.requires_grad = False
                else:
                    p.requires_grad = True

        elif trainable_layers == 'output':
            print('Only training output layers')
            for i,p in enumerate(self.bert.parameters()):
                if i not in np.arange(len(nparams_list))[-8:]:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    
        elif trainable_layers == 'all':
            print('Training all layers')
            for p in self.bert.parameters():
                p.requires_grad = True
        
    
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, graph_encoder_name='gat'):
        super(Model, self).__init__()
 
        
        if graph_encoder_name == 'graph_transformer':
            self.graph_encoder = GTransEncoder(num_node_features, nout, nhid, graph_hidden_channels)
            print('Graph encoder: graph transformer')
        elif graph_encoder_name == 'gcn':
            self.graph_encoder = GCNEncoder(num_node_features, nout, nhid, graph_hidden_channels)
            print('Graph encoder: GCN')
        else:
            self.graph_encoder = GATEncoder(num_node_features, nout, nhid, graph_hidden_channels)
            print('Graph encoder: GAT')
            
        

        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder

    def load_pretrained_graph_encoder(self, path_pretrained_graph_encoder):
        
        checkpoint = torch.load(path_pretrained_graph_encoder)
        self.graph_encoder.load_state_dict(checkpoint['graph_encoder_state_dict'])

