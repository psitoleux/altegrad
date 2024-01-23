import argparse


def get_main_parser():
    parser = argparse.ArgumentParser(description="Contrastive Learning on Graph Text Dataset")

    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased",
                        help="Transformer model name (default: allenai/scibert_scivocab_uncased)")
    parser.add_argument("--nout", type=int, default=768,
                        help="Output dimension of the final layers (default: 768)")
    parser.add_argument("--num_node_features", type=int, default=300,
                        help="Number of node features (default: 300)")
    parser.add_argument("--nhid", type=int, default=300,
                        help="Hidden layer dimension (default: 300)")
    parser.add_argument("--graph_hidden_channels", type=int, default=300,
                        help="Number of channels in graph hidden layers (default: 300)")

    parser.add_argument("--trainable", type=str, default='all',
                        help="Select which layers of text encoder are trainable (default: 'all')")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size per GPU (default: 64)")
    parser.add_argument("--target_batch_size", type=int, default=64,
                        help="Targeted effective batch size after gradient accumulation (default: 64)")

    parser.add_argument("--lr", type=float, default=4e-5,
                        help="Learning rate (default: 4e-5)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("--scheduler", type=str, default='',
                        help="Learning rate scheduler (default : '')")
    parser.add_argument("--epoch_finetune", type=int, default=-1,
                        help="Epoch at which to toggle finetuning of only last layers in BERT (default: -1)")
    
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Number of epochs dedicated to warmup (default=2)")
    parser.add_argument("--nb_cycles", type=int, default=3,
                        help="Number of cycles for relevant schedulers")

    parser.add_argument("--patience", type=int, default=2,
                        help="Patience for early stopping (default: 2)")
    
    parser.add_argument("--print_every", type=int, default=50,
                        help="Print training details every X iterations (default: 50)")


    parser.add_argument("--pretrained_graph_encoder", type=str, default=None,
                        help="Path to the pretrained graph encoder state dict (optional)")

    return parser.parse_args() 


def get_pretraining_parser():
    parser = argparse.ArgumentParser(description="Graph Encoder Pretraining")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for pre-training (default: 100)")
    parser.add_argument("--val_frequency", type=int, default=1,
                        help="Frequency of evaluation on validation dataset (default: 1)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for pre-training (default: 512)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for pre-training (default: 0.01)")

    parser.add_argument("--num_node_features", type=int, default=300,
                        help="Number of node features (default: 300)")
    parser.add_argument("--nhid", type=int, default=300,
                        help="Hidden layer dimension (default: 300)")
    parser.add_argument("--graph_hidden_channels", type=int, default=300,
                        help="Number of channels in graph hidden layers (default: 300)")
    

    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Number of epochs dedicated to warmup (default=2)")
    parser.add_argument("--nb_cycles", type=int, default=3,
                        help="Number of cycles for relevant schedulers")


    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW weight decay (default : 0.01)")


    return parser.parse_args()

def get_inference_parser():
    """Create an argument parser for inference."""
    parser = argparse.ArgumentParser(description="Inference Parameters")

    parser.add_argument("-sp", "--save_path", type=str, help="Path to saved model.", default="")
    parser.add_argument("-bs", "--batch_size", type=int, help="Batch size during inference.", default=80)

    return parser.parse_args()
