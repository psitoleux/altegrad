import argparse

def get_parser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--nb_epochs', type=int, default=5)
    parser.add_argument('learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--early_stopping', type=int, default=2)
    parser.add_argument('--text_encoder', type=str, default='scibert')
    parser.add_argument('--target_batch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss', type=str, default='InfoNCE')
    parser.add_argument('--num_node_features', type=int, default=300)
    parser.add_argument('--nhid', type=int, default=300)
    parser.add_argument('--graph_hidden_channels', type=int, default=300)
    
