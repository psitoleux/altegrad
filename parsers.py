import argparse

def get_main_parser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--nb_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=4e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--early_stopping', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--loss', type=str, default='InfoNCE')

    parser.add_argument('--text_encoder', type=str, default='scibert')

    parser.add_argument('--pretraining', type=str, default='from_scratch')
    parser.add_argument('--target_batch_size', type=int, default=32)
    parser.add_argument('--num_node_features', type=int, default=300)
    parser.add_argument('--nhid', type=int, default=300)
    parser.add_argument('--graph_hidden_channels', type=int, default=300)
    
    return parser.parse_args()

def get_pretraining_parser():
    parser = argparse.ArgumentParser(description="Graph Encoder Pretraining Script")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for pre-training (default: 100)")
    parser.add_argument("--val_frequency", type=int, default=1,
                        help="Frequency of evaluation on validation dataset (default: 1)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for pre-training (default: 512)")
    parser.add_argument("--lr", type=float, default=0.02,
                        help="Learning rate for pre-training (default: 0.02)")



    parser.set_defaults(pin_mem_dl=False)

    return parser.parse_args()
